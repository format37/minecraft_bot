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
#### Requirements
* OpenAI API key
* Minecraft running server
#### Installation
```
git clone https://github.com/format37/minecraft_bot.git
cd minecraft_bot
pip3 install -r requirements.txt
```
#### How to use
* Configure the config.json
* Run the bot.py
```
python3 bot.py
```
* After spawning the bot, call like this:
```
Janet, come
```
#### Token spending
The token spending is not significant originally, because the regular messages are short and the chat history is disabled.