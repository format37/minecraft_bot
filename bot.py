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
# import pickle
# import joblib
# import dill
# import cloudpickle
# import os
# import pandas as pd
from langchain_community.llms import Ollama


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
vec3 = require('vec3')
# autoeat = require('mineflayer-auto-eat')
# autoeat = require('/home/alex/projects/minecraft_bot/node_modules/mineflayer-auto-eat/dist/index.js')
# exit()

class ConfigLoader:
    def __init__(self, config_filename='config.json'):
        self.config_filename = config_filename
        self.config = self.load_config()

        """if self.config['openai']['api_key'] == '':
            self.config['openai']['api_key'] = input("Please enter your OpenAI API key: ")""" # TODO: Enable

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
        # llm = Ollama(model="llama2")
        # llm = Ollama(model="mistral")

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
                        (self.bot_instance.bot_action_list_items, "Command to list items in Bot's inventory",
                            "Provide the name of the bot", True),
                        (self.bot_instance.bot_action_toss, "Command to toss an item stack from Bot's inventory",
                            "Provide the name of the item to toss", True),
                        (self.bot_instance.bot_action_go_sleep, "Command to go sleep in Minecraft",
                            "Provide the name of the bed to sleep", True),
                        (self.bot_instance.bot_action_find, "Command to find an item in Minecraft",
                            "Provide the name of the item to find", True),
                        (self.bot_instance.bot_action_build, "Command to build something in Minecraft",
                            "Provide the name of the item to build", True),
                      ]
                 ]
        # tools.append(DuckDuckGoSearchRun())
        # wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        # tools.append(wikipedia)
        """tools.append(
            Tool(
                args_schema=DocumentInput,
                name='Knowledge base',
                description="Providing a game information from the knowledge base",
                func=RetrievalQA.from_chain_type(llm=llm, retriever=self.retriever),
            )
        )"""
        # tools = []
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
        # self.retriever = self.document_processor.process_documents() # TODO: Enable
        
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
        # self.chat_history = [] # Chat history have no benefits yet
        # Remove messages, instead of latest 3
        if len(self.chat_history) > 3:
            self.chat_history = self.chat_history[-3:]

        chat_agent = ChatAgent(self.config, None, self)
        # chat_agent = ChatAgent(self.config, self.retriever, self) # TODO: Enable

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
                    logger.info("Player to follow is None. Stopping")
                    self.bot.pathfinder.setGoal(None)
                    return
                logger.info("Bot already near goal. Not moving")
                py_time.sleep(1)
            self.move_to_position(self.player_to_follow)
        else:
            # self.bot.chat("Goal reached.")
            logger.info("Goal reached.")

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
            logger.info(f"move_to_position Error: {e}")
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
    
    def build_map_to_vector(self, val: str, origin: vec3, level_height: int = 0) -> list:
        """
        Converts a string map representation to global coordinates based on a given origin and level height, starting from the top left corner.

        :param val: str, a map representation where '1' indicates a block placement.
        :param origin: vec3, the global reference point for the map's top left corner.
        :param level_height: int, height adjustment for building levels (defaults to 0).
        :return: list of vec3 instances representing global positions for block placement.
        """
        places = []
        map_lines = val.strip().split('\n')
        num_lines = len(map_lines)

        for i in range(num_lines):
            num_characters = len(map_lines[i])

            for j in range(num_characters):
                if map_lines[i][j] == '1':
                    x = j + origin.x
                    y = origin.y + level_height  # Y-coordinate adjusted based on the level
                    z = i + origin.z
                    places.append(vec3(x, y, z))

        return places
    
    def find_no_air_position(self, target_position):
        collizion_shift_x = 0
        collizion_shift_y = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if i == 1 and j == 1 and k == 1:
                        continue
                    block = self.bot.blockAt(target_position.add(vec3(i-1, j-1, k-1)))
                    if block.type != "air":
                        collizion_shift_x = i-1
                        collizion_shift_y = k-1
                        return collizion_shift_x, collizion_shift_y
        return collizion_shift_x, collizion_shift_y

    def bot_action_build(self, val: str) -> str:
        """
        Places blocks based on a map, building upwards from the top left corner across multiple levels.
        
        :param val: str, the block type to be placed.
        :return: str, a status message indicating the result of the operation.
        """
        logger.info(f'Building: "{val}"')

        building_shift_x = 1
        building_shift_z = 1

        # Calculate origin from bot position, considering the top left corner
        origin = vec3(
            int(self.bot.entity.position.x+building_shift_x), 
            int(self.bot.entity.position.y), 
            int(self.bot.entity.position.z+building_shift_z)
            )

        total_successful_placements = 0
        total_attempts = 0
        item_to_build = val.strip()
        build_library = {
            "Cross": {
                "cobblestone": ["""0001000""",
"""0001000""",
"""0011100""",
"""0001000""",],
"dirt": ["""0010100""",
"""0010100""",
"""0000000""",
"""0000000""",]},
            "Simple house": {
                "cobblestone": ["""0110110
0100010
0100010
0100010
0111110""",
"""0110110
0100010
0000000
0100010
0110110""",
"""0111110
0100010
0100010
0100010
0111110""",
"""0011100
0000000
0000000
0000000
0011100""",
"""0001000
0000000
0000000
0000000
0001000""",
"""0000000
0000000
0000000
0000000
0000000"""],
"glass": ["""0000000
0000000
0000000
0000000
0000000""",
"""0000000
0000000
0100010
0000000
0001000""",
"""0000000
0000000
0000000
0000000
0000000""",
"""0000000
0000000
0000000
0000000
0000000""",
"""0000000
0000000
0000000
0000000
0000000""",
"""0000000
0000000
0000000
0000000
0000000"""],
"oak_log": ["""0000000
0000000
0000000
0000000
0000000""",
"""0000000
0000000
0000000
0000000
0000000""",
"""1000001
1000001
1000001
1000001
1000001""",
"""0100010
0100010
0100010
0100010
0100010""",
"""0010100
0010100
0010100
0010100
0010100""",
"""0001000
0001000
0001000
0001000
0001000"""],
"dirt": ["""1001001
1011101
1011101
1011101
1000001""",
"""1001001
1011101
1011101
1011101
1001001""",
"""0000000
0011100
0011100
0011100
0000000""",
"""0000000
0011100
0011100
0011100
0000000""",
"""0000000
0001000
0001000
0001000
0000000""",
"""0000000
0000000
0000000
0000000
0000000"""
]
                }
        }
        if item_to_build not in build_library:
            return f'No build library for {item_to_build}'

        first_material = list(build_library[item_to_build].keys())[0]
        levels_count = len(build_library[item_to_build][first_material])
        for level in range(levels_count):
            
            for material in build_library[item_to_build]:
                # If level of the material is zeros, skip
                have_ones = False
                for i in range(len(build_library[item_to_build][material][level])):
                    if '1' in build_library[item_to_build][material][i]:
                        have_ones = True
                        break
                if not have_ones:
                    logger.info(f"Skipping level {level} for {material}")
                    continue

                # block_id = int(eval(f'mcdata.blocksByName.{material}.id'))
                # block_object = eval(f'mcdata.blocksByName.{material}')
                # Take the chosen material from the inventory
                logger.info(f"equipping: {material}")
                # self.bot.equip(block_id, 'hand')
                # item = self.bot.inventory.findInventoryItem(block_id)
                item = self.bot.inventory.findInventoryItem(material)
                if item is None:
                    # return f'No {material} in inventory'
                    # Chat to the player
                    self.bot.chat(f'No {material} in inventory. Using current item in hand: {self.bot.heldItem.name}')
                    # material = self.bot.heldItem.name
                    # continue
                self.bot.equip(item)

                level_origin = vec3(origin.x, origin.y + level, origin.z)  # Level height adjusts the y-coordinate
                logger.info(f"level_origin: {level_origin}")

                build_map = build_library[item_to_build][material][level]

                # Get global placement coordinates for the current level
                places = self.build_map_to_vector(build_map, level_origin, 0)  # No need to pass level height here, already adjusted in level_origin
                logger.info(f"places: {places}")

                successful_placements = 0
                
                for target_position in places:
                    logger.info(f"Target position for block placement: {target_position}")
                    # If hand is empty
                    if self.bot.heldItem is None:
                        self.bot.chat('I have no item in my hand')
                        # return f'No item in hand'
                        break
                    try_number = 0
                    max_tries = 4
                    placed = False
                    while not placed and try_number < max_tries:
                        logger.info(f"try: {try_number} of {max_tries}")
                        for collizion_shift_x in [-1,0,1]:
                            if placed:
                                break
                            for collizion_shift_z in [-1,0,1]:
                                if placed:
                                    break
                                if collizion_shift_x == 0 and collizion_shift_z == 0:
                                    continue



                                



                                # Moving ++
                                target_x = target_position.x+collizion_shift_x
                                target_y = target_position.y-1
                                target_z = target_position.z+collizion_shift_z                            
                                block_in_target_position = self.bot.blockAt(vec3(target_x, target_y, target_z))
                                if block_in_target_position.name == "air":
                                    logger.info(f"Skipping block in target position: {block_in_target_position.name}. Position: {target_x} {target_y} {target_z}")
                                    continue
                                else:
                                    logger.info(f"Moving to block in target position: {block_in_target_position.name}. Position: {target_x} {target_y} {target_z}")
                                try:
                                    distance_to_goal = self.bot.entity.position.distanceTo(target_position)
                                    logger.info(f"Distance to goal: {distance_to_goal}")
                                    movements = pathfinder.Movements(self.bot)
                                    self.bot.pathfinder.setMovements(movements)
                                    # collizion_shift_x = 1 if distance_to_goal == 0 else 2  # Shift to avoid collision
                                    # Find the position that is not air
                                    # collizion_shift_x, collizion_shift_z = self.find_no_air_position(target_position)
                                    # collizion_shift_x, collizion_shift_z = shift_x, shift_z

                                    """self.bot.pathfinder.setGoal(pathfinder.goals.GoalGetToBlock(
                                        target_position.x+collizion_shift_x, 
                                        origin.y, 
                                        target_position.z+collizion_shift_z
                                        ))"""
                                    # ToXY
                                    self.bot.pathfinder.setGoal(pathfinder.goals.GoalXZ(
                                        target_position.x+collizion_shift_x, 
                                        target_position.z+collizion_shift_z
                                        ))
                                    # placed = True
                                    py_time.sleep(0.5) # TODO: Check that goar reached
                                except Exception as e:
                                    logger.info(f"move_to_position Error: {e}")
                                    self.bot.chat(f"move_to_position interrupted")
                                # Moving --
                                    
                                # referenceBlock = self.bot.blockAt(target_position.subtract(vec3(0, 1, 0)))
                                new_ref_position = vec3(target_position.x+collizion_shift_x, target_position.y-1, target_position.z+collizion_shift_z)
                                referenceBlock = self.bot.blockAt(new_ref_position)
                                logger.info(f"Reference block position: {referenceBlock.position}")
                                # if referenceBlock.type == "air":  # Assuming 'air' means no block. Adjust according to your environment's representation.
                                    
                                face_vector = vec3(0, 1, 0)  # Assume placing on top

                                try:
                                    self.bot.placeBlock(referenceBlock, face_vector)
                                    successful_placements += 1
                                    placed = True
                                except Exception as e:
                                    logger.info(f"Error placing block at {target_position}: {e}")
                                    self.bot.chat(f'Try {try_number}: a to place {material} at {target_position}')
                                
                                try_number += 1

                total_successful_placements += successful_placements
                total_attempts += len(places)

        return f'Job finished with {total_successful_placements} successful placements out of {total_attempts} attempts'

    def build_map_to_vector_v0(self, val: str, origin: vec3, level_height: int = 0) -> list:
        """
        Converts a string map representation to global coordinates based on a given origin and level height.

        :param val: str, a map representation where '1' indicates a block placement.
        :param origin: vec3, the global reference point for the map.
        :param level_height: int, height adjustment for building levels (defaults to 0).
        :return: list of vec3 instances representing global positions for block placement.
        """
        places = []
        map_lines = val.strip().split('\n')
        num_lines = len(map_lines)
        num_characters = len(map_lines[0])

        for i in range(num_lines):
            for j in range(num_characters):
                if map_lines[i][j] == '1':
                    x = j - num_characters // 2 + origin.x
                    y = origin.y + level_height # Adjust y-coordinate based on the level
                    z = i - num_lines // 2 + origin.z
                    places.append(vec3(x, y, z))

        return places

    def bot_action_place_block_v0(self, val: str) -> str:
        """
        Places blocks based on a map, building upwards across multiple levels.
        :param val: str, the block type to be placed.
        :return: str, a status message indicating the result of the operation.
        """
        logger.info(f'Placing block "{val}"')

        levels = 6  # Number of levels to build
        block_id = eval(f'mcdata.blocksByName.{val.strip()}.id')
        logger.info(f"block_id: {block_id}")

        # Origin is the bot's current position at the start
        # Calculate origin from bot position using vec3 and int
        origin = vec3(int(self.bot.entity.position.x), int(self.bot.entity.position.y), int(self.bot.entity.position.z))

        total_successful_placements = 0
        total_attempts = 0

        for level in range(levels):
                
            level_origin = vec3(origin.x, origin.y, origin.z)  # Adjust origin for each level
            logger.info(f"level_origin: {level_origin}")

            # Define the map
            map = """1000001
0011100
0100010
0100010
0100010
0011100
1000001"""

            map = """0000000
0000000
0000000
0100010
0000000
0000000
0000000"""

            # Get global placement coordinates for the current level
            places = self.build_map_to_vector(map, level_origin, level)
            logger.info(f"places: {places}")

            successful_placements = 0

            for target_position in places:
                # target_position = place
                logger.info(f"Target position for block placement: {target_position}")

                referenceBlock = self.bot.blockAt(target_position.subtract(vec3(0, 1, 0)))
                face_vector = vec3(0, 1, 0)  # Assume placing on top

                try:
                    self.bot.placeBlock(referenceBlock, face_vector)
                    successful_placements += 1
                    # py_time.sleep(1) # TODO: Remove
                except Exception as e:
                    logger.info(f"Error placing block at {target_position}: {e}")
                    self.bot.chat(f'Unable to place {val}')

            total_successful_placements += successful_placements
            total_attempts += len(places)

        return f'Job finished with {total_successful_placements} successful placements out of {total_attempts} attempts'
    
    def bot_action_list_items(self, val: str) -> str:
        self.bot.chat(f"Listing items")
        """self.bot.inventory.items().forEach(item => {
            self.bot.chat(`${item.name} x ${item.count}`)
        })"""
        # self.bot.chat(f"Inventory: {self.bot.inventory.slots}")
        text = 'Inventory:'
        for item in self.bot.inventory.items():
            text += f"\n{item.count} x {item.name}"
        # self.bot.chat(f"Inventory: {self.bot.inventory.items()}")
        return text
    
    def bot_action_toss(self, val: str) -> str:
        # Search the corresponding stack
        for item in self.bot.inventory.items():
            if item.name == val:
                self.bot.tossStack(item)
                return f'Tossing {val}'
        return f'No {val} in inventory'
    
    def bot_action_go_sleep(self, val: str) -> str:
        # Search the neares bed
        self.bot.chat(f"Finding {val}")
        def on_find(err, blockPoints):
            logger.info(f"on_find call")
            if err:
                self.bot.chat(f'Error trying to find the chosen block: {err}')
            elif blockPoints:
                self.bot.chat(f'I found a {val} at {blockPoints[0].position}.')
                # Move to the block
                pos = blockPoints[0].position
                self.bot.chat(f"Trying to sleep at {pos.x} {pos.y} {pos.z}")
                try:
                    # self.bot.sleep(pos)
                    self.bot.sleep(blockPoints[0])
                    return f'Sleeping at {pos.x} {pos.y} {pos.z}'
                except Exception as e:
                    self.bot.chat(f'Unable to sleep: {e}')
                    return f'Unable to sleep: {e}'
            else:
                self.bot.chat("I couldn't find any mentioned blocks within 256.")
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
    
    def bot_action_find(self, val: str) -> str:
        self.bot.chat(f"Finding {val}")
        def on_find(err, blockPoints):
            logger.info(f"on_find call")
            if err:
                self.bot.chat(f'Error trying to find the chosen block: {err}')
                # self.bot.quit('quitting')
            elif blockPoints:
                self.bot.chat(f'I found a {val} at {blockPoints[0].position}.')
                # Move to the block
                movements = pathfinder.Movements(self.bot)
                pos = blockPoints[0].position
                logger.info(f"pos: {pos}")
                self.bot.pathfinder.setMovements(movements)
                """ https://note.com/mega_gorilla/n/n3fa94ad2ed36
                await bot.pathfinder.goto(goal); // A very useful function. This function may change your main-hand equipment.
                // Following are some Goals you can use:
                new GoalNear(x, y, z, range); // Move the bot to a block within the specified range of the specified block. `x`, `y`, `z`, and `range` are `number`
                new GoalXZ(x, z); // Useful for long-range goals that don't have a specific Y level. `x` and `z` are `number`
                new GoalGetToBlock(x, y, z); // Not get into the block, but get directly adjacent to it. Useful for fishing, farming, filling bucket, and beds. `x`, `y`, and `z` are `number`
                new GoalFollow(entity, range); // Follow the specified entity within the specified range. `entity` is `Entity`, `range` is `number`
                new GoalPlaceBlock(position, bot.world, {}); // Position the bot in order to place a block. `position` is `Vec3`
                new GoalLookAtBlock(position, bot.world, {}); // Path into a position where a blockface of the block at position is visible. `position` is `Vec3`"""
                logger.info(f"Moving to {pos.x} {pos.y} {pos.z} RANGE_GOAL: {self.RANGE_GOAL}")
                # self.bot.pathfinder.setGoal(pathfinder.goals.GoalNear(pos.x, pos.y, pos.z, self.RANGE_GOAL))
                # GoalLookAtBlock(position, bot.world, {});
                self.bot.pathfinder.setGoal(pathfinder.goals.GoalLookAtBlock(pos, self.bot.world, {}))
            else:
                self.bot.chat("I couldn't find any mentioned blocks within 256.")
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
