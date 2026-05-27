import discord
import json
import discord.sinks
from huggingface_hub import InferenceClient
from groq import Groq
import os
import asyncio
import tempfile
from discord.ext import commands
import edge_tts
import time
import re

# DC token
with open('config.json', 'r', encoding='utf-8') as f:
    conf = json.load(f)

TOKEN = conf.get("DISCORD_TOKEN")
HF_TOKEN = conf.get("HF_TOKEN")
GROQ_KEY = conf.get("GROQ_API_KEY")

MODEL = 'zai-org/GLM-4.7-Flash'

# 語音
V_NAME = "zh-TW-HsiaoChenNeural"
V_RATE = "+20%"
V_PITCH = "+0Hz"

# AI 
hf = InferenceClient(api_key=HF_TOKEN)
groq = Groq(api_key=GROQ_KEY)

sys_prompt = "你是可愛正向的損友，叫做鳩鳩，年紀大概高中生，會用不太成熟但自然不多餘(太多動作括號，例如(嘆氣))的語氣回應使用者，也不能太弱氣(例如沒有關係…(使用刪節號))，不太需要反問，有問有答就好，因為現在是在discord的語音頻道，所以字數不用太多，閒聊的話需要小於20中文字，就像是朋友(損友)聊天一樣，盡量不要中英混雜跟太多表情符號，會用自己的名字(鳩鳩)當主詞，必須完全使用台灣的繁體中文，並且不帶有AI感。"


ints = discord.Intents.default()
ints.message_content = True
ints.voice_states = True
bot = commands.Bot(command_prefix="=", intents=ints)

sessions = {}
# 防止機器人自己話還沒講完又開始講
p_lock = asyncio.Lock()
last_ch = {}

def get_sess(uid):
    if uid not in sessions:
        sessions[uid] = []
    return sessions[uid]

# TTS
async def tts(txt, path):
    c = edge_tts.Communicate(txt, V_NAME, rate=V_RATE, pitch=V_PITCH)
    await c.save(path)

# 拿回應
async def get_ai_res(uid, txt):
    hist = get_sess(uid)
    hist.append({"role": "user", "content": txt})
    msgs = [{"role": "system", "content": sys_prompt}] + hist

    def call_hf():
        try:
            gen = hf.chat_completion(
                model=MODEL, messages=msgs, max_tokens=1024, stream=True
            )
            res_txt = ""
            for c in gen:
                d = c.choices[0].delta
                if d.content:
                    res_txt += d.content
            return res_txt
        except Exception as e:
            print(f"HF error: {e}")
            raise e

    try:
        # InferenceClient 有時候會卡住 
        res = await asyncio.wait_for(asyncio.to_thread(call_hf), timeout=30.0)
        if not res:
            res = "你是在攻三曉"
    except Exception as e:
        print(f"API call failed: {e}")
        res = "鳩鳩累了，等下再聊"

    hist.append({"role": "assistant", "content": res})
    # 不然會爆 token
    if len(hist) > 20:
        sessions[uid] = hist[-20:]
    return res


# 處理語音流
class MySink(discord.sinks.WaveSink):
    def __init__(self, vc, ch, *, filters=None):
        super().__init__(filters=filters)
        self.vc = vc
        self.ch = ch
        self.last_at = time.time()
        self.silence_limit = 1.5 # 超過 1.5 秒當講完了
        self.loop = asyncio.get_running_loop()
        self.monitor = self.loop.create_task(self.check_silence())

    async def check_silence(self):
        try:
            while True:
                await asyncio.sleep(0.5)
                if not self.vc or not self.vc.is_connected():
                    break
                # 偵測靜音 自動停止錄音
                if time.time() - self.last_at > self.silence_limit:
                    if self.audio_data:
                        try:
                            self.vc.stop_recording()
                        except:
                            pass
                        break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"monitor error: {e}")

    def write(self, data, user):
        self.last_at = time.time()
        # 如果插嘴
        if self.vc.is_playing():
            self.vc.stop()
        super().write(data, user)


# 錄音完成後
async def finished_callback(sink: MySink, ch: discord.TextChannel, *args):
    if not sink.vc or not sink.vc.is_connected():
        return

    try:
        tsks = []
        for uid, data in sink.audio_data.items():
            tsks.append(handle_audio(uid, data, sink.vc, ch))
        if tsks:
            await asyncio.gather(*tsks)

        # 處理完一波後
        if sink.vc and sink.vc.is_connected():
            try:
                await asyncio.sleep(1)
                is_rec = getattr(sink.vc, 'recording', False)
                if sink.vc.is_connected() and not is_rec:
                    sink.vc.start_recording(
                        MySink(sink.vc, ch),
                        finished_callback,
                        ch
                    )
            except Exception as e:
                print(f"restart rec failed: {e}")
    except Exception as e:
        print(f"callback error: {e}")


# 語音辨識、AI 回覆、TTS 
async def handle_audio(uid, data, vc, ch):
    tmp_in = None
    try:
        raw = data.file.getvalue()
    except:
        return

    # 暫存檔
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(raw)
        tmp_in = f.name

    try:
        # STT
        with open(tmp_in, "rb") as f:
            stt_res = await asyncio.to_thread(
                groq.audio.transcriptions.create,
                file=(os.path.basename(tmp_in), f.read()),
                model="whisper-large-v3",
                language="zh",
                response_format="text"
            )
        txt = stt_res.strip()

        if not txt or not re.search(r'[\u4e00-\u9fff\w]', txt):
            return

        m = ch.guild.get_member(uid)
        if m:
            await ch.send(f"{m.display_name}：{txt}")

        # LLM
        ai_msg = await get_ai_res(uid, txt)
        print(f"鳩鳩: {ai_msg}")

        clean_msg = re.sub(r'[*#_~`>]', '', ai_msg)
        clean_msg = re.sub(r'<[^>]+>', '', clean_msg)

        tmp_out = None
        try:
            #TTS)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_out = f.name
            await tts(clean_msg, tmp_out)

            async with p_lock:
                if vc.is_connected():
                    ev = asyncio.Event()
                    def done_playing(e):
                        if tmp_out and os.path.exists(tmp_out):
                            try:
                                os.remove(tmp_out)
                            except:
                                pass
                        bot.loop.call_soon_threadsafe(ev.set)

                    try:
                        # 播放
                        vc.play(discord.FFmpegPCMAudio(tmp_out), after=done_playing)
                        await ch.send(ai_msg)
                        await asyncio.wait_for(ev.wait(), timeout=60.0)
                    except Exception as e:
                        print(f"play error: {e}")
                        if tmp_out and os.path.exists(tmp_out):
                            try:
                                os.remove(tmp_out)
                            except:
                                pass
                        ev.set()
        finally:
            if tmp_out and os.path.exists(tmp_out):
                await asyncio.sleep(1)
                try:
                    os.remove(tmp_out)
                except:
                    pass

    except Exception as e:
        print(f"process error: {e}")
    finally:
        if tmp_in and os.path.exists(tmp_in):
            try:
                os.remove(tmp_in)
            except:
                pass


@bot.event
async def on_ready():
    print(f">> 機器人已上線：{bot.user} <<")


@bot.event
async def on_voice_state_update(m, b, a):
    # 如果頻道沒人
    vc = m.guild.voice_client
    if not vc:
        return
    if b.channel and b.channel == vc.channel:
        humans = [mem for mem in vc.channel.members if not mem.bot]
        if not humans:
            await asyncio.sleep(5) 
            humans = [mem for mem in vc.channel.members if not mem.bot]
            if not humans and vc.is_connected():
                await vc.disconnect()
                c = last_ch.get(m.guild.id)
                if c:
                    await c.send("大家都走了，那我也先閃囉！掰掰～")


@bot.command()
async def join(ctx):
    """加入頻道"""
    if not ctx.author.voice:
        return await ctx.send("你要先在語音頻道我才能進去喔")

    ch = ctx.author.voice.channel
    vc = ctx.guild.voice_client

    try:
        if vc:
            if vc.channel == ch:
                # 重連大法好
                await vc.disconnect(force=True)
                await asyncio.sleep(1)
                vc = await asyncio.wait_for(ch.connect(), timeout=20.0)
            else:
                await vc.move_to(ch)
        else:
            vc = await asyncio.wait_for(ch.connect(), timeout=20.0)

        last_ch[ctx.guild.id] = ctx.channel

        retry = 0
        while (not vc.is_connected() or vc.latency == float('inf')) and retry < 20:
            await asyncio.sleep(1)
            retry += 1

        if vc.is_connected() and vc.latency != float('inf'):
            is_rec = getattr(vc, 'recording', False)
            if not is_rec:
                await asyncio.sleep(1)
                vc.start_recording(MySink(vc, ctx.channel), finished_callback, ctx.channel)
                await ctx.send(f"進來了！我在 {ch.name} 聽你說話～")
        else:
            await ctx.send("連線怪怪的")

    except Exception as e:
        print(f"join error: {e}")
        await ctx.send(f"進不去：{e}")


@bot.command()
async def ai(ctx, *, prompt: str):
    """測試 AI"""
    res = await get_ai_res(ctx.author.id, prompt)
    await ctx.send(res)


@bot.command()
async def leave(ctx):
    """滾出頻道"""
    vc = ctx.guild.voice_client
    if vc:
        try:
            try:
                vc.stop_recording()
            except:
                pass
            await vc.disconnect(force=True)
            await ctx.send("下次聊！")
        except:
            await ctx.guild.change_voice_state(channel=None)
            await ctx.send("強制離開了")
    else:
        await ctx.send("我不在語音頻道裡啊")


@bot.event
async def on_message(msg):
    if msg.author.bot:
        return
    await bot.process_commands(msg)


async def main():
    async with bot:
        await bot.start(TOKEN)

if __name__ == "__main__":
    asyncio.run(main())
