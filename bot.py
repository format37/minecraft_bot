# https://api.python.langchain.com/en/latest/agents/langchain.agents.tool_calling_agent.base.create_tool_calling_agent.html
import os
from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
from langchain.agents import Tool, initialize_agent
# from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from pydantic.v1 import BaseModel, Field
# from langchain.tools.base import StructuredTool
from langchain_core.tools import StructuredTool
from langchain.schema import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from javascript import require, On
import json
import logging
import time as py_time
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel as BaseModel_v2
from openai import OpenAI
from typing import List, Optional, Dict
from openscad_to_voxels import generate_blueprint_from_description, print_object3d
import asyncio
# vec3 = require('vec3')

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

class ConfigLoader:
    def __init__(self, config_filename='config.json'):
        self.config_filename = config_filename
        self.config = self.load_config()

    def load_config(self):
        with open(self.config_filename, 'r') as file:
            return json.load(file)
        
temp_config = ConfigLoader('config.json').config
mcdata = require('minecraft-data')(temp_config['minecraft']['version'])

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

class BotActionType(BaseModel):
    # val: str = Field(default="", description="Corresponding parameter", required=True)
    val: str = Field(description="Corresponding parameter")

class ChatAgent:
    def __init__(self, config, retriever, bot_instance):
        self.config = config
        # Read model from json file
        keys_filename = "keys.json"
        with open(keys_filename, "r") as f:
            self.keys = json.load(f)
        os.environ["LANGCHAIN_API_KEY"] = self.keys["LANGCHAIN_API_KEY"]
        self.retriever = retriever
        self.bot_instance = bot_instance  # Passing the Bot instance to the ChatAgent
        # logger.info(f"ChatAgent function: {self.bot_instance.bot_action_come}")
        self.agent = None
        self.agent_executor = None
        # logger.info(f"Calling agent_initialization")
        # await self.agent_initialization()

    async def run_agent(self, user_input, chat_history):
        return await self.agent_executor.ainvoke(
            {
                "input": user_input,
                "chat_history": chat_history,
            }
        )
    
    async def initialize(self):
        logger.info(f"Calling agent_initialization")
        await self.agent_initialization()

    async def agent_initialization(self):
        logger.info(f">> agent_initialization")
        if self.config['langchain']['llm'] == "openai":
            openai_config = self.config['llms']['openai']
            # logger.info(f"initialize_agent with key: {self.keys['OPENAI_API_KEY']}, model: {openai_config['model']}")
            # max_tokens=3000
            llm = ChatOpenAI(
                openai_api_key=self.keys["OPENAI_API_KEY"],
                model=openai_config['model'],
                temperature=openai_config['temperature'],
            )
        if self.config['langchain']['llm'] == "anthropic":
            llm = ChatAnthropic(
                model=self.config['llms']['anthropic']['model'],
                api_key=self.keys["ANTHROPIC_API_KEY"],
                temperature=self.config['llms']['anthropic']['temperature'],
                )
        # ollama structed_chat sample
        # llm = Ollama(model="mistral:v0.3")
        # mistral:v0.3
        # llm = ChatOpenAI(
        #     api_key="ollama",
        #     model="llama3",
        #     base_url="http://localhost:11434/v1",
        # )

        # Read built_prompt_description from a file
        # with open('build_prompt_description.txt', 'r') as file:
        #     build_prompt_description = file.read()

        if self.config['langchain']['agent_type'] =="tool_calling_agent":
            return_direct = False
        elif self.config['langchain']['agent_type'] =="structed_chat":
            return_direct = True

        # tools = [self.create_structured_tool(func, name, description, return_direct)
        #          for func, name, description, return_direct in [
        #                 # (self.bot_instance.bot_action_come, "Command to come to Minecraft player",
        #                 #     "Provide the name of the player asking to come", return_direct),
        #                 # (self.bot_instance.bot_action_follow, "Command to follow for Minecraft player",
        #                 #     "Provide the name of the player asking to follow", return_direct),
        #                 # (self.bot_instance.bot_action_stop, "Command to stop performing any actions in Minecraft",
        #                 #     "You may provide the name of player asking to stop", return_direct),
        #                 (self.bot_instance.bot_action_take, "Command to take an item in Minecraft",
        #                     "Provide the name of the item to take", return_direct),
        #                 (self.bot_instance.bot_action_list_items, "Command to list items in Bot's inventory",
        #                     "Provide the name of the bot", return_direct),
        #                 (self.bot_instance.bot_action_toss, "Command to toss an item stack from Bot's inventory",
        #                     "Provide the name of the item to toss", return_direct),
        #                 (self.bot_instance.bot_action_go_sleep, "Command to go sleep in Minecraft",
        #                     "Provide the name of the bed to sleep", return_direct),
        #                 (self.bot_instance.bot_action_find, "Command to find an item in Minecraft",
        #                     "Provide the name of the item to find", return_direct),
        #                 # (self.bot_instance.bot_action_build, "Command to build anything in Minecraft",
        #                 #     build_prompt_description, return_direct),
        #               ]
        #          ]
        tools = []
        bot_action_build_tool = StructuredTool.from_function(
            coroutine=self.bot_instance.bot_action_build,
            # func=self.bot_instance.bot_action_build,
            name="bot_action_build",
            description="Command to build anything in Minecraft",
            args_schema=self.bot_instance.bot_action_build_args,
            verbose=True,
        )        
        tools.append(bot_action_build_tool)

        bot_action_come_tool = StructuredTool.from_function(
            coroutine=self.bot_instance.bot_action_come,
            # func=self.bot_instance.bot_action_come,
            name="bot_action_come",
            description="Command to come to Minecraft player",
            args_schema=self.bot_instance.bot_action_come_args,
            verbose=True,
        )        
        tools.append(bot_action_come_tool)

        bot_action_follow_tool = StructuredTool.from_function(
            coroutine=self.bot_instance.bot_action_follow,
            # func=self.bot_instance.bot_action_follow,
            name="bot_action_follow",
            description="Command to follow for Minecraft player",
            args_schema=self.bot_instance.bot_action_come_args,
            verbose=True,
        )
        tools.append(bot_action_follow_tool)

        bot_action_stop_tool = StructuredTool.from_function(
            coroutine=self.bot_instance.bot_action_stop,
            # func=self.bot_instance.bot_action_stop,
            name="bot_action_stop",
            description="Command to stop performing any actions in Minecraft",
            args_schema=self.bot_instance.bot_action_empty_args,
            verbose=True,
        )
        tools.append(bot_action_stop_tool)

        # bot_action_take_tool = StructuredTool.from_function(
        #     coroutine=self.bot_instance.bot_action_take,
        #     # func=self.bot_instance.bot_action_take,
        #     name="bot_action_take",
        #     description="Command to take an item in Minecraft",
        #     args_schema=self.bot_instance.bot_action_take_args,
        #     verbose=True,
        # )
        # tools.append(bot_action_take_tool)

        bot_action_list_items_tool = StructuredTool.from_function(
            coroutine=self.bot_instance.bot_action_list_items,
            # func=self.bot_instance.bot_action_list_items,
            name="bot_action_list_items",
            description="Command to list items in Bot's inventory",
            args_schema=self.bot_instance.bot_action_empty_args,
            verbose=True,
        )
        tools.append(bot_action_list_items_tool)

        bot_action_toss_tool = StructuredTool.from_function(
            coroutine=self.bot_instance.bot_action_toss,
            # func=self.bot_instance.bot_action_toss,
            name="bot_action_toss",
            description="Command to toss an item stack from Bot's inventory",
            args_schema=self.bot_instance.bot_action_toss_args,
            verbose=True,
        )
        tools.append(bot_action_toss_tool)

        bot_action_go_sleep_tool = StructuredTool.from_function(
            coroutine=self.bot_instance.bot_action_go_sleep,
            # func=self.bot_instance.bot_action_go_sleep,
            name="bot_action_go_sleep",
            description="Command to go sleep in Minecraft",
            args_schema=self.bot_instance.bot_action_go_sleep_args,
            verbose=True,
        )
        tools.append(bot_action_go_sleep_tool)

        bot_action_find_tool = StructuredTool.from_function(
            coroutine=self.bot_instance.bot_action_find,
            # func=self.bot_instance.bot_action_find,
            name="bot_action_find",
            description="Command to find an item in Minecraft",
            args_schema=self.bot_instance.bot_action_find_args,
            verbose=True,
        )
        tools.append(bot_action_find_tool)

        if self.config['langchain']['agent_type'] =="tool_calling_agent":
            logger.info(f"initialize_agent with tool_calling_agent")
            # prompt = ChatPromptTemplate.from_messages(
            #     [
            #         ("system", "You are a Minecraft player"),
            #         ("placeholder", "{chat_history}"),
            #         ("human", "{input}"),
            #         ("placeholder", "{agent_scratchpad}"),
            #     ]
            # )
            system_message = SystemMessagePromptTemplate.from_template(
                """You are a Minecraft player."""
            )
            human_message = HumanMessagePromptTemplate.from_template("{input}")
            ai_message = AIMessagePromptTemplate.from_template("{agent_scratchpad}")
            prompt = ChatPromptTemplate.from_messages([
                system_message,
                human_message,
                ai_message
            ])
            logger.info(f"Prompt: {prompt}")
            agent = create_tool_calling_agent(llm, tools, prompt)
            self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        if self.config['langchain']['agent_type'] =="structed_chat":
            logger.info(f"initialize_agent with structed_chat")
            self.agent = initialize_agent(
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
        self.chat_agent = ChatAgent(self.config, None, self)
        # self.retriever = self.document_processor.process_documents() # TODO: Enable
        # chat_agent = ChatAgent(self.config, self.retriever, self) # TODO: Enable
        
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

        self.loop = asyncio.get_event_loop()
        self.setup_event_handlers()

    async def initialize(self):
        await self.chat_agent.initialize()

    # Bind event handlers to the bot instance
    def setup_event_handlers(self):
        @On(self.bot, 'spawn')
        def handle_spawn(*args):
            self.loop.create_task(self.handle_spawn(*args))

        @On(self.bot, 'chat')
        def handle_message(this, sender, message, *args):
            self.loop.create_task(self.handle_message(this, sender, message, *args))

        @On(self.bot, "goal_reached")
        def handle_goal_reached(*args):
            self.loop.create_task(self.handle_goal_reached(*args))

        @On(self.bot, 'end')
        def handle_end(*args):
            self.loop.create_task(self.handle_end(*args))
        # @On(self.bot, 'spawn')
        # def handle_spawn(*args):
        #     self.handle_spawn(*args)
        # @On(self.bot, 'chat')
        # async def handle_message(this, sender, message, *args):
        #     await self.handle_message(this, sender, message, *args)
        # @On(self.bot, "goal_reached")
        # def handle_goal_reached(*args):
        #     self.handle_goal_reached(*args)
        # @On(self.bot, 'end')
        # def handle_end(*args):
        #     self.handle_end(*args)
        logger.info(f"BOT function: {self.bot_action_come}")

    # building ++
    def place_block(self, block_name, position):
        block_name = block_name.lower()
        # Check if the target position is already occupied
        target_block = self.bot.blockAt(position)
        if target_block.name != "air":
            self.bot.chat(f'Block {target_block.name} already exists at {position}')
            return False

        # Find the face to place against
        faces = [
            vec3(0, -1, 0),  # bottom
            vec3(0, 1, 0),   # top
            vec3(1, 0, 0),   # east
            vec3(-1, 0, 0),  # west
            vec3(0, 0, 1),   # south
            vec3(0, 0, -1)   # north
        ]
        
        for face in faces:
            neighbor_pos = position.plus(face)
            neighbor_block = self.bot.blockAt(neighbor_pos)
            if neighbor_block.name != "air":
                try:
                    # Make sure we're holding the correct block
                    self.equip_block(block_name)

                    # Place the block
                    self.bot.placeBlock(neighbor_block, face.scaled(-1))
                    
                    # Wait a short time for the block to be placed
                    py_time.sleep(0.1)
                    
                    # Verify that the block was placed
                    placed_block = self.bot.blockAt(position)
                    if placed_block.name == block_name:
                        # self.bot.chat(f'Successfully placed {block_name} at {position}')
                        return True
                    else:
                        self.bot.chat(f'Failed to place {block_name} at {position}. Found {placed_block.name} instead.')
                        return False
                except Exception as e:
                    self.bot.chat(f'Error while placing {block_name} at {position}: {e}')
        
        self.bot.chat(f'No suitable face found to place {block_name} at {position}')
        return False
    
    class bot_action_build_args(BaseModel):
        description: str = Field(description="The description of the required object.")
        material: str = Field(description="Material. For example, cobblestone.")
        size_x: int = Field(description="Target size of the X side of the object. For example, 10")
        size_y: int = Field(description="Target size of the Y side of the object. For example, 10")
        size_z: int = Field(description="Target size of the Z side of the object. For example, 10")
        steps: int = Field(description="Steps. For example, 6. Each step can improve quality but will increase the price and time to generate the object.")

    async def bot_action_build(self, description: str, material: str, size_x: int, size_z: int, size_y: int, steps: int) -> str:
        # Say: Let me think..
        self.bot.chat(f"Let me think, how to build {description} of size {size_x}x{size_y}x{size_z} with {material} material. Count of steps: {steps}")
        # blueprint = self.generate_blueprint_from_description(description)
        # variant = blueprint.variants[-1]
        obj = generate_blueprint_from_description(
            user_request = description, 
            size_x = size_x,
            size_y = size_y,
            size_z = size_z,
            steps_limit = steps
        )
        if obj is not None:
            obj.material = material
        print_object3d(obj)
        self.bot.chat(f'Okay, I will build this. {obj.description}')
        
        ground_level = self.determine_ground_level()
        origin = vec3(
            int(self.bot.entity.position.x),
            ground_level,
            int(self.bot.entity.position.z)
        )
        
        # Ensure the bot is facing north
        self.bot.lookAt(vec3(origin.x, origin.y, origin.z - 1))
        
        total_successful_placements = 0
        total_attempts = 0
        
        # for obj in variant.objects:
            # for y, layer in enumerate(reversed(obj.layers)):  # Reverse the order of layers
        for y, layer in enumerate(obj.layers):
            for z, row in enumerate(layer.rows):
                for x, block in enumerate(row.blocks):
                    if block == 1:
                        abs_position = vec3(
                            origin.x + x,
                            origin.y + y,
                            origin.z + z
                        )
                        
                        # Check if a block already exists at the target position
                        existing_block = self.bot.blockAt(abs_position)
                        if existing_block.name != "air":
                            self.bot.chat(f'Block {existing_block.name} already exists at {abs_position}')
                            continue

                        if not self.check_support(abs_position):
                            self.add_support(abs_position)
                        
                        # Find a suitable position for the bot to stand
                        bot_position = self.find_suitable_position(abs_position)
                        if bot_position:
                            self.move_to_block_position(bot_position)
                            
                            # Wait for the bot to reach the position
                            py_time.sleep(0.5)
                            
                            # Ensure the bot is looking at the placement position
                            self.bot.lookAt(abs_position)
                            
                            success = self.place_block(obj.material, abs_position)
                            # Wait a short time for the block to be placed
                            py_time.sleep(0.1)
                            
                            if success:
                                total_successful_placements += 1
                            total_attempts += 1
                        else:
                            self.bot.chat(f"Unable to find a suitable position to place block at {abs_position}")
        
        return f'Job finished with {total_successful_placements} successful placements out of {total_attempts} attempts'

    def find_suitable_position(self, target_position):
        # Check positions around and above the target block
        adjacent_offsets = [
            vec3(0, 1, 0),   # above
            vec3(1, 0, 0),   # east
            vec3(-1, 0, 0),  # west
            vec3(0, 0, 1),   # south
            vec3(0, 0, -1),  # north
            vec3(1, 0, 1),   # southeast
            vec3(1, 0, -1),  # northeast
            vec3(-1, 0, 1),  # southwest
            vec3(-1, 0, -1)  # northwest
        ]
        
        for offset in adjacent_offsets:
            check_pos = target_position.plus(offset)
            block = self.bot.blockAt(check_pos)
            
            # If the block is solid (not air) and the block above it is air (so the bot can stand there)
            if block.name != "air" and self.bot.blockAt(check_pos.offset(0, 1, 0)).name == "air":
                # Check if there's enough headroom (2 blocks of air)
                if self.bot.blockAt(check_pos.offset(0, 2, 0)).name == "air":
                    # logger.info(f"Found suitable position for bot to stand at {check_pos}")
                    return check_pos.offset(0, 1, 0)  # Return the position on top of the solid block
        
        # If no suitable position is found above, check positions at the same level
        same_level_offsets = [
            vec3(1, 0, 0),   # east
            vec3(-1, 0, 0),  # west
            vec3(0, 0, 1),   # south
            vec3(0, 0, -1),  # north
            vec3(1, 0, 1),   # southeast
            vec3(1, 0, -1),  # northeast
            vec3(-1, 0, 1),  # southwest
            vec3(-1, 0, -1)  # northwest
        ]
        
        for offset in same_level_offsets:
            check_pos = target_position.plus(offset)
            block = self.bot.blockAt(check_pos)
            
            # If the block is air and the block below it is solid
            if block.name == "air" and self.bot.blockAt(check_pos.offset(0, -1, 0)).name != "air":
                logger.info(f"Found suitable position for bot to stand at {check_pos}")
                return check_pos
        
        # If no suitable position is found, return None
        logger.info(f"No suitable position found for bot to stand near {target_position}")
        return None

    def move_to_block_position(self, position):
        movements = pathfinder.Movements(self.bot)
        self.bot.pathfinder.setMovements(movements)
        goal = pathfinder.goals.GoalNear(position.x, position.y, position.z, 0)  # Set range to 0 to ensure the bot is exactly at the position
        self.bot.pathfinder.setGoal(goal)
        # self.bot.once('goal_reached', lambda: None)
        self.bot.once('goal_reached', lambda *args: None)

    def equip_block(self, block_name):
        item = self.bot.inventory.findInventoryItem(block_name)
        if item:
            try:
                self.bot.equip(item, 'hand')
            except Exception as e:
                self.bot.chat(f'Error equipping {block_name}: {e}')
        else:
            self.bot.chat(f'No {block_name} found in inventory')

#     def generate_blueprint_from_description_(self, description: str) -> Blueprint:
#         # Set up OpenAI client
#         os.environ["OPENAI_API_KEY"] = self.chat_agent.keys['OPENAI_API_KEY']
#         client = OpenAI()

#         # Define the system message
#         system_message = """
# You are an AI assistant specialized in creating Minecraft blueprints. 
# Given a description, you should generate a detailed blueprint for a Minecraft structure.
# Your response should be a valid Blueprint object as defined by the provided structure.
# You need to generate 3 versions of the blueprint. Each verstion should be more complex than the previous one.

# Important guidelines for blueprint generation:
# 1. Avoid floating structures: Ensure that each block above the ground level is supported by at least one block beneath it.
# 2. Build from the bottom up: First layer is bottom layer, second layer is above the first, and so on.
# 3. Use 0 for air/empty space and 1 for solid blocks.
# 4. Each layer should have the same dimensions (number of rows and blocks per row).
# 5. Consider structural integrity: Large open spaces should have supporting pillars or walls.

# Example of a proper layer structure for a 3x3 building:
# Layer 0 (Ground): [[1,1,1],[1,0,1],[1,1,1]]
# Layer 2 (Walls):  [[1,1,1],[1,0,1],[1,1,1]]
# Layer 3 (Roof):   [[1,1,1],[1,1,1],[1,1,1]]

# Ensure that your blueprint follows these guidelines for all variants.
# """

#         # Make the API call
#         try:
#             completion = client.beta.chat.completions.parse(
#                 model="gpt-4o-2024-08-06",
#                 messages=[
#                     {"role": "system", "content": system_message},
#                     {"role": "user", "content": f"Create a Minecraft blueprint for: {description}"},
#                 ],
#                 response_format=Blueprint,
#                 max_tokens=16383,
#                 temperature=1.0
#             )

#             message = completion.choices[0].message
#             if message.parsed:
#                 logger.info(f"Generated blueprints: {message.parsed}")
#                 return message.parsed
#             else:
#                 raise ValueError("Failed to generate blueprint: " + str(message.refusal))
#         except Exception as e:
#             print(f"An error occurred: {e}")
#             # Return a dummy blueprint in case of an error
#             return Blueprint(
#                 name="Error Blueprint",
#                 description="An error occurred while generating the blueprint.",
#                 variants=[
#                     BlueprintVariant(
#                         name="Error Variant",
#                         objects=[
#                             Object3D(
#                                 layers=[Layer(rows=[Row(blocks=[1])])],
#                                 material="stone"
#                             )
#                         ],
#                         description="This is a placeholder due to an error in blueprint generation."
#                     )
#                 ]
#             )

    def determine_ground_level(self) -> int:
        bot_position = self.bot.entity.position
        for y in range(int(bot_position.y), -64, -1):  # Assuming -64 is the lowest possible y-coordinate
            block = self.bot.blockAt(vec3(bot_position.x, y, bot_position.z))
            if block.name != "air":
                return y + 1  # Return the y-coordinate above the first non-air block
        return 0  # Fallback to y=0 if no ground is found

    def check_support(self, position: vec3) -> bool:
        below = self.bot.blockAt(position.offset(0, -1, 0))
        return below.name != "air"

    def add_support(self, position: vec3):
        support_position = position.offset(0, -1, 0)
        self.place_block("dirt", support_position)
    # building --

    async def handle_spawn(self, *args):
        logger.info("I spawned.")

    async def handle_message(self, this, sender, message, *args):
        logger.info(f"Sender: {sender} Message: {message}")
        if not self.config['minecraft']['username'] in message or \
            sender == self.config['minecraft']['username']:
            logger.info("Not a message for me")
            return
        # self.chat_history = [] # Chat history have no benefits yet
        # Remove messages, instead of latest 3
        if len(self.chat_history) > 3:
            self.chat_history = self.chat_history[-3:]

        user_input = message
        logger.info(f'user_input: {user_input}')
        user_input = f"Player: {sender}. Message: {message}"
        logger.info(f'sending:\n{user_input}')
        
        if self.config['langchain']['agent_type'] == "tool_calling_agent":
            response = await self.chat_agent.run_agent(user_input, self.chat_history)
        elif self.config['langchain']['agent_type'] == "structed_chat":
            response = await self.chat_agent.agent.arun(input=user_input, chat_history=self.chat_history)
        else:
            response = 'No agent type specified in config.cfg ["tool_calling_agent", "structed_chat"]'
        # elif self.config['langchain']['agent_type'] =="structed_chat":
        #     # try:
        #     response = self.chat_agent.agent.run(input=user_input, chat_history=self.chat_history)
        #     # except Exception as e:
        #     #     logger.info(f"Error: {e}")
        #     #     response = f"Error: {e}"
        # else:
        #     response = 'No agent type specified in config.cfg ["tool_calling_agent", "structed_chat"]'
        
        logger.info(f'response:\n{response}')
        self.chat_history.append(HumanMessage(content=message))
        # self.chat_history.append(AIMessage(content=response))
        response_output = response["output"]
        logger.info(f"response_output: {response_output}")
        self.chat_history.append(AIMessage(content=str(response_output)))
        self.bot.chat(response_output)

    async def handle_end(self, *args):
        logger.info("Bot ended")

    async def handle_goal_reached(self, *args):
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

    async def move_to_position(self, sender):
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
            self.bot.pathfinder.goto(pathfinder.goals.GoalNear(pos.x, pos.y, pos.z, self.RANGE_GOAL))
        except Exception as e:
            logger.info(f"move_to_position Error: {e}")
            self.bot.chat(f"move_to_position interrupted")

    class bot_action_come_args(BaseModel):
        val: str = Field(description="Player name to come to")
    
    async def bot_action_come(self, val: str) -> str:
        await self.move_to_position(val)
        return f'Moving to {val}'

    async def bot_action_follow(self, val: str) -> str:
        self.player_to_follow = val
        await self.move_to_position(val)
        return f'Following to {val}'
    
    class bot_action_empty_args(BaseModel):
        pass

    async def bot_action_stop(self) -> str:
        self.player_to_follow = None
        self.bot.pathfinder.setGoal(None)        
        return f'I have stopped'
    
    class bot_action_take_args(BaseModel):
        val: str = Field(description="Item name to take")
    
    async def bot_action_take(self, val: str) -> str:
        self.bot.equip(val)
        return f'Equipping {val}'
    
    def build_map_to_vector(self, val: list, origin: vec3, level_height: int = 0) -> list:
        places = []
        num_lines = len(val)
        for i in range(num_lines):
            num_characters = len(val[i])
            for j in range(num_characters):
                if val[i][j] == '1':
                    x = j + origin.x
                    y = origin.y + level_height
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
    
    def add_dirt_support(self, blueprint):
        if 'dirt' not in blueprint:
            first_key = list(blueprint.keys())[0]
            num_layers = len(blueprint[first_key])
            num_lines = len(blueprint[first_key][0])
            width = len(blueprint[first_key][0][0])
            
            # Invert all values in all materials
            for material in blueprint:
                for layer in range(num_layers):
                    for line in range(num_lines):
                        logger.info(f"material: {material}, layer: {layer}, line: {line}")
                        new_line = ''.join('0' if c == '1' else '1' for c in blueprint[material][layer][line])
                        logger.info(f"new_line: {new_line}")
                        blueprint[material][layer][line] = new_line
            
            # Add dirt filled with ones
            blueprint['dirt'] = [['1' * width] * num_lines for _ in range(num_layers)]
            
            # Multiply each material table onto the dirt table
            for material in blueprint:
                if material == 'dirt':
                    continue
                for layer in range(num_layers):
                    for line in range(num_lines):
                        blueprint['dirt'][layer][line] = ''.join(str(int(a) * int(b)) for a, b in zip(blueprint['dirt'][layer][line], blueprint[material][layer][line]))
            
            # Convert all materials except dirt back to one
            for material in blueprint:
                if material == 'dirt':
                    continue
                for layer in range(num_layers):
                    for line in range(num_lines):
                        blueprint[material][layer][line] = ''.join('1' if c == '0' else '0' for c in blueprint[material][layer][line])
        
        return blueprint
    
    async def bot_action_list_items(self) -> str:
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
    
    class bot_action_toss_args(BaseModel):
        val: str = Field(description="Item name to toss")
    
    async def bot_action_toss(self, val: str) -> str:
        # Search the corresponding stack
        for item in self.bot.inventory.items():
            if item.name == val:
                self.bot.tossStack(item)
                return f'Tossing {val}'
        return f'No {val} in inventory'
    
    class bot_action_go_sleep_args(BaseModel):
        val: str = Field(description="Block name to sleep")
    
    async def bot_action_go_sleep(self, val: str) -> str:
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
    
    class bot_action_find_args(BaseModel):
        val: str = Field(description="Block name to find")
    
    async def bot_action_find(self, val: str) -> str:
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

async def main():
    # os.environ['REQ_TIMEOUT'] = '400000'
    # config = ConfigLoader('config.json').config
    # chat_agent = ChatAgent(config, None, None)
    # chat_history = []
    # chat_history.append(HumanMessage(content='Hello, my name Alex, I need to create a bid.'))
    # chat_history.append(AIMessage(content='Hello, Alex! Which date?'))
    # user_input = '2024-05-16'
    # response = chat_agent.agent_executor.invoke(
    #     {
    #         "input": user_input,
    #         "chat_history": chat_history,
    #     }
    # )
    # logger.info(f'response:\n{response}')
    bot_instance = Bot()
    await bot_instance.initialize()
    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
