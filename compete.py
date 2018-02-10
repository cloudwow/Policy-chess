import tensorflow as tf
from arena import Arena
from bot import Bot
from tree_bot import TreeBot
from dumbbot import DumbBot
from random_bot import RandomBot
sess = tf.InteractiveSession()

player_1 = Bot(sess, "p1", 3, 3)
player_2 = TreeBot(sess, "p2")
#player_2 = DumbBot(2)  #Bot(sess, "p2")
#player_2 = RandomBot()
sess.run(tf.initialize_all_variables())

arena = Arena(sess, player_1, player_2)

arena.play(1000)
