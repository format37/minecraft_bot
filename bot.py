import json
import logging
from pydantic import BaseModel, Field
from langchain.agents import Tool, initialize_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.tools import StructuredTool
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from javascript import require, On
from langchain.chains import RetrievalQA
import time as py_time
# from langchain.globals import set_debug
# set_debug(True)


# Define the info logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
log_format = '%(asctime)s - %(levelname)s - %(message)s'
date_format = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(log_format, date_format)
handler.setFormatter(formatter)
logger.addHandler(handler)

mineflayer = require('mineflayer')
pathfinder = require('mineflayer-pathfinder')
blockfinder = require('mineflayer-blockfinder')(mineflayer)

class ConfigLoader:
    def __init__(self, config_filename='config.json'):
        self.config_filename = config_filename
        self.config = self.load_config()

        if self.config['openai']['api_key'] == '':
            self.config['openai']['api_key'] = input("Please enter your OpenAI API key: ")

    def load_config(self):
        with open(self.config_filename, 'r') as file:
            return json.load(file)

temp_config = ConfigLoader('config.json').config
mcdata = require('minecraft-data')(temp_config['minecraft']['version'])

# Save block list to a file
"""with open(f"{temp_config['context']['path']}/blocks.txt", 'w') as file:
    for block in mcdata.blocksArray:
        file.write(block.name + '\n')"""

class DocumentProcessor:
    def __init__(self, config):
        self.config = config

    def process_documents(self):
        context_path = self.config['context']['path']
        loader = DirectoryLoader(context_path, glob="*", loader_cls=TextLoader)
        docs = loader.load()
        embeddings = OpenAIEmbeddings(openai_api_key=self.config['openai']['api_key'])
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(docs)
        vector = DocArrayInMemorySearch.from_documents(documents, embeddings)
        print(f"### Vector processed!")
        return vector.as_retriever()


class DocumentInput(BaseModel):
    question: str = Field()


class ChatAgent:
    def __init__(self, config, retriever, bot_instance):
        self.config = config
        self.retriever = retriever
        self.bot_instance = bot_instance  # Passing the Bot instance to the ChatAgent
        logger.info(f"ChatAgent function: {self.bot_instance.bot_action_come}")
        self.agent = self.initialize_agent()

    def initialize_agent(self):
        llm = ChatOpenAI(
            openai_api_key=self.config['openai']['api_key'],
            model=self.config['openai']['model'],
            temperature=self.config['openai']['temperature']
        )

        tools = [self.create_structured_tool(func, name, description, return_direct)
                 for func, name, description, return_direct in [
                        (self.bot_instance.bot_action_come, "Command to come to Minecraft player",
                            "Provide the name of the player asking to come", True),
                        (self.bot_instance.bot_action_follow, "Command to follow for Minecraft player",
                            "Provide the name of the player asking to follow", True),
                        (self.bot_instance.bot_action_stop, "Command to stop performing any actions in Minecraft",
                            "You may provide the name of player asking to stop", True),
                        (self.bot_instance.bot_action_take, "Command to take an item in Minecraft",
                            "Provide the name of the item to take", True),
                        (self.bot_instance.bot_action_find, "Command to find an item in Minecraft",
                            "Provide the name of the item to find", True)
                      ]
                 ]
        tools.append(DuckDuckGoSearchRun())
        # wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        # tools.append(wikipedia)
        tools.append(
            Tool(
                args_schema=DocumentInput,
                name='Knowledge base',
                description="Providing a game information from the knowledge base",
                func=RetrievalQA.from_chain_type(llm=llm, retriever=self.retriever),
            )
        )

        return initialize_agent(
            tools,
            llm,
            agent='chat-conversational-react-description',
            verbose=True,
            handle_parsing_errors=True
        )

    @staticmethod
    def create_structured_tool(func, name, description, return_direct):
        logger.info(f"create_structured_tool name: {name} func: {func}")
        return StructuredTool.from_function(
            func=func,
            name=name,
            description=description,
            args_schema=BotActionType,
            return_direct=return_direct,
        )


class Bot:
    RANGE_GOAL = 1

    def __init__(self):
        self.player_to_follow = None
        self.chat_history = []
        self.config = ConfigLoader('config.json').config
        self.document_processor = DocumentProcessor(self.config)
        self.retriever = self.document_processor.process_documents()
        
        self.bot = mineflayer.createBot({
            'host': self.config['minecraft']['host'],
            'port': self.config['minecraft']['port'],
            'username': self.config['minecraft']['username'],
        })
        if not self.bot._client.version == self.config['minecraft']['version']:
            print(f"Bot version: {self.bot._client.version} is not equal to config version: {self.config['minecraft']['version']}")
        self.bot.loadPlugin(pathfinder.pathfinder)
        self.bot.loadPlugin(blockfinder)
        logger.info("Started mineflayer")
        # Bind event handlers to the bot instance
        @On(self.bot, 'spawn')
        def handle_spawn(*args):
            self.handle_spawn(*args)
        @On(self.bot, 'chat')
        def handle_message(this, sender, message, *args):
            self.handle_message(this, sender, message, *args)
        @On(self.bot, "goal_reached")
        def handle_goal_reached(*args):
            self.handle_goal_reached(*args)
        @On(self.bot, 'end')
        def handle_end(*args):
            self.handle_end(*args)
        logger.info(f"BOT function: {self.bot_action_come}")

    def handle_spawn(self, *args):
        logger.info("I spawned.")

    def handle_message(self, this, sender, message, *args):
        logger.info(f"Sender: {sender} Message: {message}")
        if not self.config['minecraft']['username'] in message or \
            sender == self.config['minecraft']['username']:
            logger.info("Not a message for me")
            return
        self.chat_history = [] # Chat history have no benefits yet

        chat_agent = ChatAgent(self.config, self.retriever, self)

        user_input = f"Player: {sender}. Message: {message}"
        logger.info(f'sending:\n{user_input}')
        response = chat_agent.agent.run(input=user_input, chat_history=self.chat_history)

        self.chat_history.append(HumanMessage(content=message))
        self.chat_history.append(AIMessage(content=response))
        self.bot.chat(response)

    def handle_end(self, *args):
        logger.info("Bot ended")

    def handle_goal_reached(self, *args):
        if self.player_to_follow is not None:
            player_position = self.bot.players[self.player_to_follow].entity.position
            while self.bot.entity.position.distanceTo(player_position) <= self.RANGE_GOAL * 2:
                if self.player_to_follow is None:
                    return
                logger.info("Bot already near goal. Not moving")
                py_time.sleep(1)
            self.move_to_position(self.player_to_follow)
        else:
            self.bot.chat("Goal reached.")

    def move_to_position(self, sender):
        try:
            player = self.bot.players[sender]
            target = player.entity
            if not target:
                self.bot.chat("I don't see you !")
                return
            movements = pathfinder.Movements(self.bot)
            pos = target.position
            self.bot.pathfinder.setMovements(movements)
            logger.info(f"Moving to {pos.x} {pos.y} {pos.z} RANGE_GOAL: {self.RANGE_GOAL}")
            self.bot.pathfinder.setGoal(pathfinder.goals.GoalNear(pos.x, pos.y, pos.z, self.RANGE_GOAL))
        except Exception as e:
            logger.error(f"move_to_position Error: {e}")
            self.bot.chat(f"move_to_position interrupted")

    def bot_action_come(self, val: str) -> str:
        self.move_to_position(val)
        return f'Moving to {val}'

    def bot_action_follow(self, val: str) -> str:
        self.player_to_follow = val
        self.move_to_position(val)
        return f'Following to {val}'

    def bot_action_stop(self, val: str) -> str:
        self.player_to_follow = None
        self.bot.pathfinder.setGoal(None)        
        return f'I have stopped'
    
    def bot_action_take(self, val: str) -> str:
        self.bot.equip(val)
        return f'Equipping {val}'
    
    def bot_action_find(self, val: str) -> str:
        self.bot.chat(f"Finding {val}")
        def on_find(err, blockPoints):
            logger.info(f"on_find call")
            if err:
                self.bot.chat(f'Error trying to find the chosen block: {err}')
                # self.bot.quit('quitting')
            elif blockPoints:
                self.bot.chat(f'I found a mentioned block at {blockPoints[0].position}.')
                # Move to the block
                self.bot.pathfinder.setMovements(pathfinder.Movements(self.bot))
                self.bot.pathfinder.setGoal(pathfinder.goals.GoalBlock(blockPoints[0].position.x, blockPoints[0].position.y, blockPoints[0].position.z))
            else:
                self.bot.chat("I couldn't find any mentioned blocks within 256.")
                # self.bot.quit('quitting')
        # Check if the block exists
        """if not hasattr(mcdata.blocksByName, val):
            message = f'There is no block named {val}.'
            self.bot.chat(message)
            return message"""

            # self.bot.quit('quitting')
        try:
            block_id = eval(f'mcdata.blocksByName.{val}.id')
        except Exception as e:
            message = f'There is no block named {val}.'
            self.bot.chat(message)
            return message
        
        self.bot.chat(f'Block id {block_id}')
        self.bot.findBlock({
            'point': self.bot.entity.position,
            'matching': block_id,
            'maxDistance': 256,
            'count': 1
        }, on_find)

        return f'Finding {val}'


class BotActionType(BaseModel):
    val: str = Field(description="Player name")

if __name__ == "__main__":
    bot_instance = Bot()
