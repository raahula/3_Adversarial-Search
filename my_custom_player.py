
from sample_players import DataPlayer
import random, math
import time
#import numpy as np
from collections import defaultdict
from abc import ABC, abstractmethod

BUFFER_TIME = .005
time_limit = .150 - BUFFER_TIME

class CustomPlayer_alpha_beta(DataPlayer):
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
        
        import random
        if state.ply_count<2:
            self.queue.put(random.choice(state.actions()))
        else:
            for d in range (1, 100):
                self.queue.put(self.alpha_beta_search(state, d))

    def min_value(self, gameState, alpha, beta, depth):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """
        if gameState.terminal_test():
            return gameState.utility(0)

        if depth<=0:
            return self.score(gameState)
    
        v = float("inf")
        for a in gameState.actions():
            v = min(v, self.max_value(gameState.result(a), alpha, beta, depth-1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    def alpha_beta_search(self, gameState, depth):
        """ Return the move along a branch of the game tree that
        has the best possible value.  A move is a pair of coordinates
        in (column, row) order corresponding to a legal move for
        the searching player.
        
        You can ignore the special case of calling this function
        from a terminal state.
        """
        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_move = None
        
        for a in gameState.actions():
            v = self.min_value(gameState.result(a), alpha, beta, depth-1)
            alpha = max(alpha, v)
            if v > best_score:
                best_score = v
                best_move = a
        return best_move



    # TODO: modify the function signature to accept an alpha and beta parameter
    def max_value(self, gameState, alpha, beta, depth):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """
        if gameState.terminal_test():
            return gameState.utility(0)

        if depth<=0:
            return self.score(gameState)
    
        v = float("-inf")
        for a in gameState.actions():
            v = max(v, self.min_value(gameState.result(a), alpha, beta, depth-1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

'''          
    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)
'''



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

        #use this for simulation count 
        '''
        for i in range(0, 10):           
            v = self.node_selection()
            player_won = v.simulation()
            #print ("winner is player {} for iteration {}".format(player_won, i))
            v.backpropagate(player_won)
            #print(i)
            #print (v.backpropagate(player_won))
            #print(self.root.best_child(c_parameter = 0.5))
        return self.root.best_child(c_parameter = 0.5).whatactionwasperformedfcs
        '''   

        #use this for time 
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
            #if current_node.visit_count<10:
            #    return random.choice(current_node.children)
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

            '''
        li = self.untried_action()
        l = len(li)
        return l == 0
        '''

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
        #weight = [(c.score_parent_perspective/ c.visit_count) + c_parameter*math.sqrt((2*math.log(self.visit_count)/c.visit_count)) \
        #for c in self.children]
        #return self.children[np.argmax(weight)]
        return ch
  

    def score_parent_perspective(self):
        wins = self._result[self.parent.state.player()]
        losses = self._result[1-self.parent.state.player()]

        #wins = self._result[0]
        #losses = self._result[1]

        #print(wins - losses)
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