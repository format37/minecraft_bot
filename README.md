# Minecraft bot
Mineflayer Langchain Agent  
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
* Janet, build stone Cross
* Janet, build Simple cobblestone house
#### Requirements
* Docker
* Langsmith api key (or you have to disable the langsmith)
* Minecraft running server with port 33733 (configurable)
#### Instalation
```
git clone https://github.com/format37/minecraft_bot.git
cd minecraft_bot
```
#### Run
* Configure keys.json
* Configure config.json
```
docker compose up --build
```
* After spawning the bot, call like this:
```
Janet, come to YourUserName
```
#### Token spending
The token spending is not significant originally, because the regular messages are short and the chat history is disabled.