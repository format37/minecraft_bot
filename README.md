# Minecraft bot
Mineflayer Langchain ReAct Agent  
Based on [mineflayer](https://github.com/PrismarineJS/mineflayer/tree/master/examples/python) and [langchain](https://python.langchain.com)
#### Commands examples
* Janet, come to me
* Janet, follow me  
* Janet, stop  
* Janet, what the weather tomorrow in London?  
* Janet, i am looking for iron_ore  
* Janet, sleep at purple_bed
* Janet, search in the knowledge base the coordinates of home
* Janet, what items do you have in your inventory?
* Janet, toss dirt
* Janet, build Cross
* Janet, build Simple house
#### Requirements
* Minecraft running server with port 33733
#### Installation
```
conda create --name minecraft python=3.9.16
conda activate minecraft
git clone https://github.com/format37/minecraft_bot.git
cd minecraft_bot
pip3 install -r requirements.txt
```
#### How to use
* Configure the config.json
* Define your timeout to avoid [JSPyBridge](https://github.com/extremeheat/JSPyBridge) limit.
```
export REQ_TIMEOUT=400000
```
* Run bot.py
```
python3 bot.py
```
* After spawning the bot, call like this:
```
Janet, come
```
#### Token spending
The token spending is not significant originally, because the regular messages are short and the chat history is disabled.