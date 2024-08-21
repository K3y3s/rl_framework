



class TestEnv():
    
    """
    Board:
        0, 1
        2, 3
        
    Starting at 0. 
    If 3 end of game with reward 1.
    If 2 moving to 0 with reward 0.
    
    """
    
    def __init__(self):
        self.actions = [0, #left
                        1, #right
                        2, #down
                        3 #up
                        ]
        
        self.states = [0,1,2,3]
        self.current_state = 0
        
    
    def reset(self):
        self.current_state = 0

        return self.current_state, 0
    
    def step(self, action:int):
        
        if self.current_state == 0:
            
            if action == 0:
                self.current_state == 0
                terminate = False
                reward = 0
                
            elif action == 1 or action ==3:
                self.current_state == 1
                terminate = False
                reward = 0
        
            else:
                self.current_state == 2
                terminate = True
                reward = 0
        
        elif self.current_state == 1:
             
            if action == 0 or action == 3:
                self.current_state == 0
                terminate = False
                reward = 0
                
            elif action == 1:
                self.current_state == 1
                terminate = False
                reward = 0
        
            else:
                self.current_state == 3
                terminate = True
                reward = 1
                
                
        return self.current_state, reward, terminate, None, None