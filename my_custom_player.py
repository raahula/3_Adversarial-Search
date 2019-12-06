
from sample_players import DataPlayer
import random, math
import time
#import numpy as np
from collections import defaultdict
from abc import ABC, abstractmethod

BUFFER_TIME = .005
time_limit = .150 - BUFFER_TIME


class CustomPlayer_mcts(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
       
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
               
        
        '''
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            self.queue.put(self.best_action(state))
        '''    
        self.queue.put(self.best_action(state))

        
    def best_action(self, state):
        self.root = monteCarloNode(state, whatactionwasperformedfcs= None, parent=None)       

        start_time = time.time()

        while time.time() - start_time < time_limit:           
            v = self.node_selection()
            player_won = v.simulation()
            v.backpropagate(player_won)
        if self.root.children:
            return self.root.best_child(c_parameter = 0.5).whatactionwasperformedfcs
        else:
            return random.choice(self.root.state.actions)
        
        
    def node_selection(self):
        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.all_actions_explored():
                return current_node.try_unexplored_action()

            else:
                current_node = current_node.best_child()
        return current_node

class monteCarloNode():
    def __init__(self, state, whatactionwasperformedfcs, parent = None):
        self._visits = 0
        self._result = defaultdict(int)
        self._untried_action = None
        self.children = []
        self.state = state
        self.parent = parent
        self.whatactionwasperformedfcs = whatactionwasperformedfcs

    def all_actions_explored(self):
        if not self.untried_action():
            return True

    def try_unexplored_action(self):
        action = self.untried_action().pop()
        next_state = self.state.result(action)
        child_node = monteCarloNode(next_state, whatactionwasperformedfcs= action, parent = self)
        self.children.append(child_node)
        return child_node
    
   
    def untried_action(self):
        if self._untried_action is None:
            self._untried_action = self.state.actions() #available legal actions
        return self._untried_action

    def best_child(self, c_parameter=1.4):
        w = float("-inf")
        ch = None
        for c in self.children:
            weight = (c.score_parent_perspective()/ c.visit_count()) + c_parameter*math.sqrt((2*math.log(self.visit_count())/c.visit_count()))
            if weight > w:
                w=weight
                ch = c
        return ch
  

    def score_parent_perspective(self):
        wins = self._result[self.parent.state.player()]
        losses = self._result[1-self.parent.state.player()]
        return wins

    def visit_count(self):
        return self._visits

    def is_terminal_node(self):
        return self.state.terminal_test()

    def rollout_policy(self, possible_moves):
        return random.choice(possible_moves)

    def simulation(self):
        current_rollout_state = self.state
        player = self.state.player()
        while not current_rollout_state.terminal_test():
            possible_moves = current_rollout_state.actions()
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.result(action)
            if current_rollout_state.utility(player) == float("inf"):
                return player
            if current_rollout_state.utility(player) == float("-inf"):
                return (1-player)
            if current_rollout_state.utility(player) == 0:
                return ("draw")        

    def backpropagate(self, player_won):
        self._visits+=1
        if player_won!="draw":
            self._result[player_won] += 2
        else:
            self._result[0] += 0.5
            self._result[1] += 0.5
        if self.parent:
            self.parent.backpropagate(player_won)

CustomPlayer = CustomPlayer_mcts