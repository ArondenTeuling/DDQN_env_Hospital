
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 10:50:23 2021

@author: s149562
"""
import pandas as pd
from datetime import datetime
import numpy as np
import random
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

#load data
Input_requests = pd.read_excel ('Data_set_input.xlsx')
Input_requests=Input_requests[:700]




##Model Internal Assignment Aron den Teuling



## DDQN architecture 
class DuelingDeepQNetwork(nn.Module):
    def __init__ (self, lr, input_dims, fc1_dims, fc2_dims, n_actions, chkpt_dir,name):
        super(DuelingDeepQNetwork, self).__init__()
        self.chkpt_dir=chkpt_dir
        self.checkpoint_file=os.path.join(self.chkpt_dir,name)
        self.inputs_dims=input_dims
        self.fc1_dims=fc1_dims
        self.fc2_dims=fc2_dims
        self.n_actions=n_actions
        self.fc1=nn.Linear(*self.inputs_dims, self.fc1_dims)
        self.fc2=nn.Linear(self.fc1_dims, self.fc2_dims)
        self.values=nn.Linear(self.fc2_dims, 1)
        self.actions=nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer=optim.Adam(self.parameters(), lr=lr)
        self.loss=nn.MSELoss()
        self.device=T.device('cuda:0' if T.cuda.is_available() else'cpu')
        self.to(self.device)
    
    def forward(self, state):
        x=F.relu(self.fc1(state))
        x=F.relu(self.fc2(x))
        actions=self.actions(x)
        values=self.values(x)

        return values, actions     
    
    def save_checkpoint(self):
        print ('saving a checkpoint')
        T.save(self.state_dict, self.checkpoint_file)
    
    def load_checkpoint(self):
        print('loading a checkpoint')
        self.state_dict=T.load(self.checkpoint_file)
        self.eval()
class Agent():
    def __init__(self, gamma, epsilon,lr, input_dims, batch_size, n_actions, max_mem_size=100000, eps_end=0.01, eps_dec=0.2e-5, replace=1000, chkpt_dir='dueling_ddqn_3' ):        
        self.gamma=gamma
        self.epsilon=epsilon
        self.eps_min=eps_end
        self.eps_dec=eps_dec
        self.lr=lr
        self.action_space=[i for i in range(0,n_actions)]
        self.mem_size=max_mem_size
        self.batch_size=batch_size
        self.mem_cntr=0
        self.learn_step_counter=0
        self.replace_target_cnt=replace
        self.chkpt_dir=chkpt_dir
        
        self.q_eval=DuelingDeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256, name='DuelDQN_eval', chkpt_dir=self.chkpt_dir)
        self.q_next=DuelingDeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256, name='DuelDQN_next', chkpt_dir=self.chkpt_dir)
        
        
        
        
        self.state_memory=np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory=np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory=np.zeros(self.mem_size, dtype=np.float32)
        self.reward_memory=np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory=np.zeros(self.mem_size, dtype=np.bool)
    
    
    def store_transition(self, state, action, reward, state_, done):
        index=self.mem_cntr % self.mem_size
        self.state_memory[index]=state
        self.new_state_memory[index]=state_
        self.reward_memory[index]= reward
        self.action_memory[index]= action
        self.terminal_memory[index]=done
        
        self.mem_cntr+=1
    
    def choose_action(self, observation):
        if np.random.random()> self.epsilon:
            state=T.tensor([observation]).to(self.q_eval.device)
            _, advantage=self.q_eval.forward(state)
            action=T.argmax(advantage).item()
        
        else:
            action=np.random.choice(self.action_space)
      
        return action
    
    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt ==0:
            self.q_next.load_state_dict(self.q_eval.state_dict())
    
    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()
    
    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()
            
        
    
    
    def learn(self):
        if self.mem_cntr< self.batch_size:
            return
        
        self.q_eval.optimizer.zero_grad()
        
        self.replace_target_network()
        
        
        max_mem=min(self.mem_cntr, self.mem_size)
        batch=np.random.choice(max_mem, self.batch_size, replace=False)   
        batch_index=np.arange(self.batch_size, dtype=np.float32)
        state_batch= T.tensor(self.state_memory[batch]).to(self.q_eval.device)

        new_state_batch=T.tensor(self.new_state_memory[batch]).to(self.q_eval.device)
        reward_batch=T.tensor(self.reward_memory[batch]).to(self.q_eval.device)
        terminal_batch=T.tensor(self.terminal_memory[batch]).to(self.q_eval.device)
        action_batch=self.action_memory[batch]

        V_s, A_s= self.q_eval.forward(state_batch)
        V_s_, A_s_= self.q_next.forward(new_state_batch)
        
        V_s_eval, A_s_eval= self.q_eval.forward(new_state_batch)
        
        q_pred=T.add(V_s,
                     (A_s- A_s.mean(dim=1, keepdim=True)))[batch_index, action_batch]
        
        q_next=T.add(V_s_,(A_s_-  A_s_.mean(dim=1, keepdim=True)))
        q_eval=T.add(V_s_eval,
                     (A_s_eval-  A_s_eval.mean(dim=1, keepdim=True)))
        
        max_actions=T.argmax(q_eval, dim=1)
        q_next[terminal_batch]=0.0
        
        q_target=reward_batch+self.gamma*q_next[batch_index, max_actions]
        
        loss=self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        
        self.q_eval.optimizer.step()
        self.learn_step_counter+=1
        
        
        
        self.epsilon=self.epsilon-self.eps_dec if self.epsilon> self.eps_min \
                    else self.eps_min
                    
                    
Agent=Agent(gamma=0.99, epsilon=0.05, batch_size=32, n_actions=10, eps_end=0.05, input_dims=[62], lr=0.0000 )
Agent.load_models()


#input start parameters of the system
Number_of_AGVS=50
Start_location='SPC_U1'
Start_destination='None'
BatteryLevel=100
Requests_assigned=[]
Start_Status='Idle'
Time_of_day=0
time_interval=10 ## amount of seconds you want the time interval to be
Battery_decr_driving=0.01 ##battery decrease per second when driving
Battery_decr_driving_loaded=0.015
Battery_decr_idle=0.005
Battery_level_constraint=32
Battery_increase_charging=0.04
Floors=['1' , '2', '3', '4', '5', '6', 'U1', 'U2'] # Floors for the elevator
Departements=['AHL','BEV', 'KU', 'KVB', 'LAB', 'KU', 'NEV','GAS', 'ADM', 'SPC', 'MTFS']
time=23300

n=0



class AGV:
    
    def __init__(self, ID,  Current_location, Destination_location, BatteryLevel, Requests_assigned, Status):
        self.ID=ID
        self.Current_location= Current_location
        self.Destination_location=Destination_location
        self.BatteryLevel=BatteryLevel
        self.Requests_assigned=Requests_assigned
        self.Status=Status
        self.Total_Travel_time_current_task=0
        self.Travel_time_done=0
        self.Elevator_time_done=0
        self.Elevator_time_current_Task=0
        self.Request_serving='None'
        self.Dispatching_location='None'
    
    def Idle(self):
        if len(self.Requests_assigned)==0 and self.BatteryLevel>32:
            self.Status='Driving_to_dispatch_location'
            Dispatching_DQN.Add_step(self.Current_location)
            Dispatch_location=list_departements[Dispatching_DQN.action]+'_U1'
            Simulator.reward=0
            self.Dispatching_location=Dispatch_location
            self.Total_Travel_time_current_task=Time_calculator(self.Current_location, self.Dispatching_location)
            self.Elevator_time_current_Task=Lift_move_time_calculator(Simulator.time,self.Current_location, self.Dispatching_location)
#            print('At time is '+ str(Simulator.time)+ ', AGV :'  +  str(self.ID)+ ' is driving to dispatch location: '+ self.Dispatching_location)
        if self.BatteryLevel<=32:
            self.Status='Driving_to_Charging'
            nearest_charging_station=Charging_station_finder(self.Current_location)
            self.Destination_location=nearest_charging_station
            self.Total_Travel_time_current_task=Time_calculator(self.Current_location, self.Destination_location)
            self.Elevator_time_current_Task=Lift_move_time_calculator(Simulator.time, self.Current_location, self.Destination_location)
#            print('At time is '+ str(Simulator.time)+ ', AGV :'  +  str(self.ID)+ ' is driving to charging location: '+ self.Destination_location)
        if len(self.Requests_assigned)>0 and self.BatteryLevel>32:
            self.Status='Driving_to_request'
            self.Request_serving=self.Requests_assigned[0]
            self.Destination_location=Request_dict_simulation[self.Requests_assigned[0]].Pick_up_location
            Picking_up_time=np.random.normal(350, 50)
            self.Total_Travel_time_current_task=Time_calculator(self.Current_location, self.Destination_location)+Picking_up_time
            self.Elevator_time_current_Task=Lift_move_time_calculator(Simulator.time, self.Current_location, self.Destination_location)
#            print('At time is '+ str(Simulator.time) + ', AGV : ' + str(self.ID)  +' is driving to request number: ' + str(self.Requests_assigned[0]))
            
        
    
            
        

    
    def Drive_to_request(self):
            if self.Travel_time_done<self.Total_Travel_time_current_task: 
                self.BatteryLevel-=time_interval*Battery_decr_driving
                self.Travel_time_done+=time_interval
            if (self.Elevator_time_current_Task> self.Elevator_time_done) and (self.Travel_time_done>=self.Total_Travel_time_current_task):
                self.Elevator_time_done+=time_interval
                self.BatteryLevel-=time_interval*Battery_decr_idle
            if (self.Elevator_time_current_Task<= self.Elevator_time_done) and (self.Travel_time_done>=self.Total_Travel_time_current_task):
                 self.Status='Transporting_request_dropoff'
                 self.Elevator_time_done=0
                 self.Travel_time_done=0
                 self.Current_location=self.Destination_location
                 self.Destination_location=Request_dict_simulation[self.Requests_assigned[0]].Destination_location
                 Request_dict_simulation[self.Requests_assigned[0]].Time_picked_up=Simulator.time
                 Reward_calculator(self.Requests_assigned[0], Simulator.time )
                 Picking_up_time=np.random.normal(350, 50)
                 self.Total_Travel_time_current_task=Time_calculator(self.Current_location, self.Destination_location)+Picking_up_time
                 self.Elevator_time_current_Task=Lift_move_time_calculator(Simulator.time, self.Current_location, self.Destination_location)
#                 print('At time is '+ str(Simulator.time) +' request number: ' + str(self.Requests_assigned[0]) + ' is picked up by AGV: ' + str(self.ID))
    
    def Transporting_request_dropoff(self):
            if self.Travel_time_done<self.Total_Travel_time_current_task: 
                self.BatteryLevel-=time_interval*Battery_decr_driving
                self.Travel_time_done+=time_interval
            if (self.Elevator_time_current_Task> self.Elevator_time_done) and (self.Travel_time_done>=self.Total_Travel_time_current_task):
                self.Elevator_time_done+=time_interval
                self.BatteryLevel-=time_interval*Battery_decr_idle
            if (self.Elevator_time_current_Task<= self.Elevator_time_done) and (self.Travel_time_done>=self.Total_Travel_time_current_task):
                 self.Elevator_time_done=0
                 self.Travel_time_done=0
                 self.Current_location=self.Destination_location
                 self.Destination_location= 'None'
#                 print('At time is '+ str(Simulator.time) +' request number: ' + str(self.Requests_assigned[0]) + ' is dropped_of up by AGV: ' + str(self.ID))
                 Request_dict_simulation[self.Requests_assigned[0]].Time_delivered=Simulator.time
                 self.Requests_assigned=self.Requests_assigned[1:]
                 if len(self.Requests_assigned)==0 and self.BatteryLevel>32:
                     self.Status='Idle'
                     self.Request_serving='None'
                 if len(self.Requests_assigned)>0 and self.BatteryLevel>32:
                     self.Status='Driving_to_request'
                     self.Request_serving=self.Requests_assigned[0]
                     self.Destination_location=Request_dict_simulation[self.Requests_assigned[0]].Pick_up_location
                     Picking_up_time=np.random.normal(350, 50)
                     self.Total_Travel_time_current_task=Time_calculator(self.Current_location, self.Destination_location)+Picking_up_time
                     self.Elevator_time_current_Task=Lift_move_time_calculator(Simulator.time, self.Current_location, self.Destination_location)
#                     print('At time is '+ str(time) + ', AGV : ' + str(self.ID)  +' is driving to request number: ' + str(self.Requests_assigned[0]))
                 if  self.BatteryLevel<=32:
                     self.Status='Driving_to_Charging'
                     nearest_charging_station=Charging_station_finder(self.Current_location)
                     self.Destination_location=nearest_charging_station
                     self.Total_Travel_time_current_task=Time_calculator(self.Current_location, self.Destination_location)
                     self.Elevator_time_current_Task=Lift_move_time_calculator(time, self.Current_location, self.Destination_location)
#                     print('At time is '+ str(Simulator.time)+ ', AGV : '  +  str(self.ID)+ ' is driving to charging location: '+ self.Destination_location)
        
    def Driving_to_Charging(self):
            if self.Travel_time_done<self.Total_Travel_time_current_task: 
                self.BatteryLevel-=time_interval*Battery_decr_driving
                self.Travel_time_done+=time_interval
            if (self.Elevator_time_current_Task> self.Elevator_time_done) and (self.Travel_time_done>=self.Total_Travel_time_current_task):
                self.Elevator_time_done+=time_interval
                self.BatteryLevel-=time_interval*Battery_decr_idle
            if (self.Elevator_time_current_Task<= self.Elevator_time_done) and (self.Travel_time_done>=self.Total_Travel_time_current_task):
                self.Current_location=self.Destination_location
                self.Destination_location='None'
                self.Status='Charging'
                self.Travel_time_done=0
                self.Total_Travel_time_current_task=0
                self.Elevator_time_current_Task=0
                self.Elevator_time_done=0
#                print('At time is '+ str(Simulator.time)+ ', AGV : '  +  str(self.ID)+ ' started_charging at : '+ self.Current_location)
                
                
    
    
    def Charging(self):
        if self.BatteryLevel<100:
            self.BatteryLevel+=time_interval*Battery_increase_charging
        if self.BatteryLevel>=100 and len(self.Requests_assigned)==0:
            self.Status='Idle'
        if self.BatteryLevel>32 and len(self.Requests_assigned)>0:
            self.Status='Driving_to_request'
            self.Request_serving=self.Requests_assigned[0]
            self.Destination_location=Request_dict_simulation[self.Requests_assigned[0]].Pick_up_location
            Picking_up_time=np.random.normal(350, 50)
            self.Total_Travel_time_current_task=Time_calculator(self.Current_location, self.Destination_location)+Picking_up_time
            self.Elevator_time_current_Task=Lift_move_time_calculator(Simulator.time, self.Current_location, self.Destination_location)
#            print('At time is '+ str(Simulator.time) + ', AGV : ' + str(self.ID)  +' is driving to request number: ' + str(self.Requests_assigned[0]))

    def Driving_to_dispatch_location(self):
            if self.Travel_time_done<self.Total_Travel_time_current_task: 
                self.BatteryLevel-=time_interval*Battery_decr_driving
                self.Travel_time_done+=time_interval
            if (self.Elevator_time_current_Task> self.Elevator_time_done) and (self.Travel_time_done>=self.Total_Travel_time_current_task):
                self.Elevator_time_done+=time_interval
                self.BatteryLevel-=time_interval*Battery_decr_idle
            if (self.Elevator_time_current_Task<= self.Elevator_time_done) and (self.Travel_time_done>=self.Total_Travel_time_current_task):
                self.Travel_time_done=0
                self.Total_Travel_time_current_task=0
                self.Elevator_time_current_Task=0
                self.Elevator_time_done=0
                self.Current_location=self.Dispatching_location
                self.Status='Dispatched'
                self.Dispatching_location='None'
#                print('At time is '+ str(Simulator.time)+ ', AGV : '  +  str(self.ID)+ ' is dispatched at : '+ self.Current_location)
    
    
    def Dispatched(self):
        if self.BatteryLevel<100:
            self.BatteryLevel+=time_interval*Battery_increase_charging
        if len(self.Requests_assigned)>0 and self.BatteryLevel>32:
             self.Status='Driving_to_request'
             self.Request_serving=self.Requests_assigned[0]
             self.Destination_location=Request_dict_simulation[self.Requests_assigned[0]].Pick_up_location
             Picking_up_time=np.random.normal(350, 50)
             self.Total_Travel_time_current_task=Time_calculator(self.Current_location, self.Destination_location)+Picking_up_time
             self.Elevator_time_current_Task=Lift_move_time_calculator(Simulator.time, self.Current_location, self.Destination_location)
#             print('At time is '+ str(Simulator.time) + ', AGV : ' + str(self.ID)  +' is driving to request number: ' + str(self.Requests_assigned[0]))


            


class Request:
    def __init__(self, Pick_up_location, Destination_location, Time_arrival, Time_picked_up, Time_delivered):
        self.Pick_up_location=Pick_up_location
        self.Destination_location=Destination_location
        self.Time_arrival=Time_arrival
        self.Time_picked_up=Time_picked_up
        self.Time_delivered=Time_delivered


     


## Make multiple AGV objects and puts them in a dictionairy, to easily retrieve them during the simulation
AGV_list=[]
for x in range(0, Number_of_AGVS):
    AGV_list.append("AGV_"+str(x))

AGV_classes_information={}
for name in AGV_list:
        AGV_classes_information[name]=AGV(name, Start_location, Start_destination, BatteryLevel, Requests_assigned, Start_Status)


Request_dict_simulation={}


def Request_generator(Date_simulation, Time_simulation):
    if Date_simulation>0:
        start_range=when_days_end[Date_simulation]
    else:
        start_range=0
    end_range=when_days_end[Date_simulation+1]
    New_request_list=[]
    for x in range (start_range,end_range):
        if Input_requests.Date[x]==Date_simulation:
            if (Time_simulation-time_interval)<=Input_requests.Time[x]<(Time_simulation):
                if  ('PAS' not in Input_requests.Pick_up_location[x]) and ('PAS' not in Input_requests.Drop_off_location[x]): 
                    New_request_list.append(x)
                    Request_dict_simulation[x]=Request(Input_requests.Pick_up_location[x], Input_requests.Drop_off_location[x], Time_simulation, 'None', 'None')
    return New_request_list
    

def Converter_to_seconds(Input_requests) :
    for x in range (0, len(Input_requests)):
        date_time=Input_requests.Time[x]
        totalseconds=(date_time.hour*3600)+(date_time.minute*60)+(date_time.second)
        Input_requests.Time[x]=totalseconds
        
    return (Input_requests)

def Convert_date_to_day(Input_requests) :
    list_of_days=[]
    for x in range (0, len(Input_requests)):
        if Input_requests.Date[x] not in list_of_days:
            list_of_days.append(Input_requests.Date[x])
            Input_requests.Date[x]=list_of_days.index(Input_requests.Date[x])
        elif Input_requests.Date[x] in list_of_days:
            Input_requests.Date[x]=list_of_days.index(Input_requests.Date[x])
    return Input_requests

def Location_counter(Input_requests):
    list_locations=[]
    for x in range(0, len(Input_requests)):
        if Input_requests.Pick_up_location[x] not in list_locations:
            list_locations.append( Input_requests.Pick_up_location[x])
        if Input_requests.Drop_off_location[x] not in list_locations: 
            list_locations.append(Input_requests.Drop_off_location[x])
    return(list_locations)

when_days_end=[]    
def Day_slicer_input_requests():
    day=0
    for x in range(0, len(Input_requests)):
        if Input_requests.Date[x]==day:
            when_days_end.append(x)
            day=day+1
            
     
            
            
        
def Transform_locations(Input_requests):
    for x in range (0, len(Input_requests)):
        location=Input_requests.Pick_up_location[x]
        Outputlocation=''
        if 'AHL' in location:
            Outputlocation='AHL'
        if 'BEV' in location:
            Outputlocation='BEV'
        if 'KU' in location:
            Outputlocation='KU'
        if 'KVB' in location:
            Outputlocation='KVB'
        if 'PAS' in location:
            Outputlocation='PAS'
        if 'LAB' in location:
            Outputlocation='LAB'
        if 'KU' in location:
            Outputlocation='KU'
        if 'NEV' in location:
            Outputlocation='NEV'
        if 'GAS'in location:
            Outputlocation='GAS'
        if 'ADM'in location:
            Outputlocation='ADM'
        if 'SPC'in location:
            Outputlocation='SPC'
        if 'MTFS' in location:
            Outputlocation='MTFS'
        if 'U1' in location:
            Outputlocation+='_U1'
        if 'U2' in location:
            Outputlocation+='_U2'
        if '01' in location:
            Outputlocation+='_1'
        if '02' in location:
            Outputlocation+='_2'
        if '2H' in location:
            Outputlocation+='_2'
        if '03'in location:
            Outputlocation+='_3'
        if '04'in location:
            Outputlocation+='_4'
        if '05'in location:
            Outputlocation+='_5'
        if '06'in location:
            Outputlocation+='_6'
        Input_requests.Pick_up_location[x]=Outputlocation
    for x in range (0, len(Input_requests)):
        location=Input_requests.Drop_off_location[x]
        Outputlocation=''
        if 'AHL' in location:
            Outputlocation='AHL'
        if 'BEV' in location:
            Outputlocation='BEV'
        if 'KU' in location:
            Outputlocation='KU'
        if 'KVB' in location:
            Outputlocation='KVB'
        if 'PAS' in location:
            Outputlocation='PAS'
        if 'LAB' in location:
            Outputlocation='LAB'
        if 'KU' in location:
            Outputlocation='KU'
        if 'NEV' in location:
            Outputlocation='NEV'
        if 'GAS'in location:
            Outputlocation='GAS'
        if 'ADM'in location:
            Outputlocation='ADM'
        if 'SPC'in location:
            Outputlocation='SPC'
        if 'MTFS' in location:
            Outputlocation='MTFS'
        if 'U1' in location:
            Outputlocation+='_U1'
        if 'U2' in location:
            Outputlocation+='_U2'
        if '01' in location:
            Outputlocation+='_1'
        if '02' in location:
            Outputlocation+='_2'
        if '2H' in location:
            Outputlocation+='_2'
        if '03'in location:
            Outputlocation+='_3'
        if '04'in location:
            Outputlocation+='_4'
        if '05'in location:
            Outputlocation+='_5'
        if '06'in location:
            Outputlocation+='_6'
        Input_requests.Drop_off_location[x]=Outputlocation
        
    return (Input_requests)

## Here the functions transforms the input data to usefull data for the simulation
list_locations=Location_counter(Input_requests)
Converter_to_seconds(Input_requests)
Convert_date_to_day(Input_requests)
Transform_locations(Input_requests)




        
list_locations=[ 'AHL_6', 'AHL_5', 'AHL_4', 'AHL_3', 'AHL_2', 'AHL_1', 'AHL_U1', 'AHL_U2', 'BEV_6', 'BEV_5', 'BEV_4', 'BEV_3', 'BEV_2', 'BEV_1', 'BEV_U1', 
                'NEV_6', 'NEV_5', 'NEV_4', 'NEV_3', 'NEV_2', 'NEV_1', 'NEV_U1', 'GAS_6', 'GAS_5', 'GAS_4', 'GAS_3', 'GAS_2', 'GAS_1', 'GAS_U1',
                'KVB_6', 'KVB_5', 'KVB_4', 'KVB_3', 'KVB_2', 'KVB_1', 'KVB_U1', 'SPC_U1', 'KU_6', 'KU_5', 'KU_2', 'KU_1', 'KU_U1', 'LAB_5', 'LAB_4', 'LAB_3',
                'LAB_2', 'LAB_1', 'LAB_U1', 'ADM_U1', 'MTFS_U1']

list_departements=[ 'AHL', 'SPC', 'GAS', 'KU', 'NEV', 'BEV', 'KVB', 'LAB', 'ADM', 'MTFS']

def Time_matrix():
    Time_matrix_departements=[]
    Time_matrix_departements_AHL=[0, np.random.normal(145,35),np.random.normal(70,10), (np.random.normal(70,10)+np.random.normal(35,10)),(np.random.normal(70,10)+np.random.normal(35,10)+np.random.normal(155,60) ),np.random.normal(70,20), (np.random.normal(70,20)+np.random.normal(120,45)),(np.random.normal(70,20)+np.random.normal(120,45)+np.random.normal(30,10)), (np.random.normal(30,10)+np.random.normal(145,35)), (np.random.normal(70,20)+np.random.normal(120,45)+np.random.normal(100,35))   ]
    Time_matrix_departements_AHL=[i*1.5 for i in Time_matrix_departements_AHL]
    Time_matrix_departements.append(Time_matrix_departements_AHL)
    Time_matrix_departements_SPC=[ np.random.normal(145,35),0,(np.random.normal(145,35)+np.random.normal(70,10)),(np.random.normal(145,35)+np.random.normal(70,10)+np.random.normal(35,10)),(np.random.normal(30,10)+np.random.normal(125,50)+np.random.normal(30,10)+np.random.normal(100,35)),(np.random.normal(70,20)+np.random.normal(145,35)),(np.random.normal(30,10)+np.random.normal(125,50)+np.random.normal(30,10) ),(np.random.normal(30,10)+np.random.normal(125,50)), np.random.normal(30,10), (np.random.normal(30,10)+np.random.normal(125,50)+np.random.normal(30,10) +np.random.normal(100,35)  )     ]
    Time_matrix_departements_SPC=[i*1.5 for i in Time_matrix_departements_SPC]
    Time_matrix_departements.append(Time_matrix_departements_SPC)
    Time_matrix_departements_GAS=[np.random.normal(70,10),(np.random.normal(145,35)+np.random.normal(70,10)),0,np.random.normal(35,10),(np.random.normal(35,10)+np.random.normal(155,60)),(np.random.normal(70,20)+np.random.normal(120,45)),(np.random.normal(120,45)+np.random.normal(70,20)+np.random.normal(70,10)),(np.random.normal(120,45)+np.random.normal(70,20)+np.random.normal(70,10)+np.random.normal(30,10)), (np.random.normal(30,10)+np.random.normal(145,35)+np.random.normal(70,10)), (np.random.normal(120,45)+np.random.normal(70,20)+np.random.normal(70,10)+np.random.normal(100,35))  ]
    Time_matrix_departements_GAS=[i*1.5 for i in Time_matrix_departements_GAS]
    Time_matrix_departements.append(Time_matrix_departements_GAS)
    Time_matrix_departements_KU=[(np.random.normal(70,10)+np.random.normal(35,10)),(np.random.normal(145,35)+np.random.normal(70,10)+np.random.normal(35,10)),np.random.normal(35,10),0, np.random.normal(155,60), (np.random.normal(70,20)+np.random.normal(70,10)+np.random.normal(35,10)), (np.random.normal(100,35)+np.random.normal(155,60)+np.random.normal(100,35)),(np.random.normal(155,60)+np.random.normal(100,35)+np.random.normal(30,10)), (np.random.normal(30,10)+np.random.normal(145,35)+np.random.normal(70,10)+np.random.normal(35,10)), (np.random.normal(100,35)+np.random.normal(100,35)+ np.random.normal(155,60))  ]
    Time_matrix_departements_KU=[i*1.5 for i in Time_matrix_departements_KU]
    Time_matrix_departements.append(Time_matrix_departements_KU)
    Time_matrix_departements_NEV=[(np.random.normal(70,10)+np.random.normal(35,10)+np.random.normal(155,60) ),(np.random.normal(30,10)+np.random.normal(125,50)+np.random.normal(30,10)+np.random.normal(100,35)),(np.random.normal(35,10)+np.random.normal(155,60)),np.random.normal(155,60),0, (np.random.normal(120,45)+np.random.normal(100,35)), np.random.normal(100,35), (np.random.normal(100,35)+np.random.normal(30,10)),(np.random.normal(100,35)+np.random.normal(30,10)+ np.random.normal(125,50)), np.random.normal(100,35)   ]
    Time_matrix_departements_NEV=[i*1.5 for i in Time_matrix_departements_NEV]
    Time_matrix_departements.append(Time_matrix_departements_NEV)
    Time_matrix_departements_BEV=[np.random.normal(70,20),(np.random.normal(70,20)+np.random.normal(145,35)),(np.random.normal(70,20)+np.random.normal(70,10)),(np.random.normal(70,20)+np.random.normal(70,10)+np.random.normal(35,10)),(np.random.normal(120,45)+np.random.normal(100,35)),0 , np.random.normal(120,45),  (np.random.normal(120,45)+ np.random.normal(30,10)),(np.random.normal(30,10)+ np.random.normal(145,35)+ np.random.normal(70,20)),(np.random.normal(100,35)+ np.random.normal(120,45))    ]
    Time_matrix_departements_BEV=[i*1.5 for i in Time_matrix_departements_BEV]
    Time_matrix_departements.append(Time_matrix_departements_BEV)
    Time_matrix_departements_KVB=[Time_matrix_departements[0][6],Time_matrix_departements[1][6],Time_matrix_departements[2][6],Time_matrix_departements[3][6],Time_matrix_departements[4][6], Time_matrix_departements[5][6],0, np.random.normal(30,10),(np.random.normal(30,10)+np.random.normal(125,50)),np.random.normal(100,35)   ]
    Time_matrix_departements_KVB=[i*1.5 for i in Time_matrix_departements_KVB]
    Time_matrix_departements.append(Time_matrix_departements_KVB)
    Time_matrix_departements_LAB=[Time_matrix_departements[0][7],Time_matrix_departements[1][7],Time_matrix_departements[2][7],Time_matrix_departements[3][7],Time_matrix_departements[4][7], Time_matrix_departements[5][7],Time_matrix_departements[6][7],0 ,np.random.normal(125,50),(np.random.normal(100,35)+np.random.normal(30,10))  ]
    Time_matrix_departements_LAB=[i*1.5 for i in Time_matrix_departements_LAB]
    Time_matrix_departements.append(Time_matrix_departements_LAB)
    Time_matrix_departements_ADM=[Time_matrix_departements[0][8],Time_matrix_departements[1][8],Time_matrix_departements[2][8],Time_matrix_departements[3][8],Time_matrix_departements[4][8], Time_matrix_departements[5][8],Time_matrix_departements[6][8],Time_matrix_departements[7][8],0, (np.random.normal(125,50)+np.random.normal(30,10)+np.random.normal(100,35))]
    Time_matrix_departements_ADM=[i*1.5 for i in Time_matrix_departements_ADM]
    Time_matrix_departements.append(Time_matrix_departements_ADM)
    Time_matrix_departements_MTFS=[Time_matrix_departements[0][9],Time_matrix_departements[1][9],Time_matrix_departements[2][9],Time_matrix_departements[3][9],Time_matrix_departements[4][9], Time_matrix_departements[5][9],Time_matrix_departements[6][9],Time_matrix_departements[7][9],Time_matrix_departements[8][9],0]
    Time_matrix_departements_MTFS=[i*1.5 for i in Time_matrix_departements_MTFS]
    Time_matrix_departements.append(Time_matrix_departements_MTFS)
    return Time_matrix_departements


def Reset_battery_when_day_over():
    for AGV in AGV_classes_information:
        AGV_classes_information[AGV].BatteryLevel=100
    

def Reset_dictionairy():
     Request_dict_simulation.clear
        

def Time_calculator(Location1, Location2):
    Location1=Location1[:Location1.index('_')]
    Location2=Location2[:Location2.index('_')]
    index_1=list_departements.index(Location1)
    index_2=list_departements.index(Location2)
    matrix= Time_matrix()
    Time_job=matrix[index_1][index_2]
    return Time_job

def Lift_move_time_calculator(time, location1, location2):
    ## hier checken of er U1 in begin of eind locatie zit anders moet hij nog een exra keer de lift
    movetime=0
    departement1=location1[:location1.index('_')]
    departement2=location2[:location2.index('_')]
    floor1=location1[location1.index('_'):]
    floor2=location2[location2.index('_'):]
    
    if departement1==departement2:
        if floor1!=floor2:
                if 23300<=time<45000:
                    movetime=np.random.normal(150, 30)
                if 45000<=time<63000:
                    movetime=np.random.normal(210, 30)
                if time>=63000:
                    movetime=np.random.normal(110, 30)
     
    if departement1!=departement2:
        if ('U1' in location1) and ('U1' in location2):
            movetime=0
        if ('U1' not in location1) and ('U1' not in location2):
            if 23300<=time<45000:
                movetime=2*np.random.normal(150, 20)
            if 45000<=time<63000:
                movetime=2*np.random.normal(210, 30)
            if time>=63000:
                movetime=2*np.random.normal(110, 30)
        if (('U1' in location1) or ('U1' in location2)) and (('U1' in location1) and ('U1' in location2))==False:
            if 23300<=time<45000:
                    movetime=np.random.normal(150, 20)
            if 45000<=time<63000:
                    movetime=np.random.normal(210, 30)
            if time>=63000:
                    movetime=np.random.normal(110, 30)
        
            
    return (movetime)
    
    



def Charging_station_finder(location):
    location=location[:location.index('_')]
    location=location+'_U1'
    return location
        
    

def Assigning_request_to_AGVS(Request): 
    time_nearest_AGV=10000000000
    Nearest_AGV='None'
    Pickup_location=Request_dict_simulation[Request].Pick_up_location
    for AGV in AGV_classes_information:
        if ((AGV_classes_information[AGV].Status=='Idle') or (AGV_classes_information[AGV].Status=='Dispatched')) and len(AGV_classes_information[AGV].Requests_assigned)== 0:
            Time_to_drive=Time_calculator(AGV_classes_information[AGV].Current_location, Pickup_location)
            Time_elevator=Lift_move_time_calculator(Simulator.time, AGV_classes_information[AGV].Current_location, Pickup_location)
            time_current_AGV_away=Time_to_drive+Time_elevator
            if time_current_AGV_away<time_nearest_AGV:
                time_nearest_AGV=time_current_AGV_away
                Nearest_AGV=AGV
    if Nearest_AGV=='None':
        for AGV in AGV_classes_information:
            if (AGV_classes_information[AGV].Status=='Charging') and AGV_classes_information[AGV].BatteryLevel>32 and len(AGV_classes_information[AGV].Requests_assigned)== 0:
                Time_to_drive=Time_calculator(AGV_classes_information[AGV].Current_location, Pickup_location)
                Time_elevator=Lift_move_time_calculator(Simulator.time, AGV_classes_information[AGV].Current_location, Pickup_location)
                time_current_AGV_away=Time_to_drive+Time_elevator
                if time_current_AGV_away<time_nearest_AGV:
                    time_nearest_AGV=time_current_AGV_away
                    Nearest_AGV=AGV
    
    
    if Nearest_AGV!='None':
        AGV_classes_information[Nearest_AGV].Requests_assigned=AGV_classes_information[Nearest_AGV].Requests_assigned+[Request]
#        print ( str(Nearest_AGV) +' is getting the request :' + str(Request)) 
    
    if Nearest_AGV== 'None':    
        return('No AGV found')
        

def State_vector_departement_maker(Location_AGV):
    number_of_dispatched_vehicles_per_location=[0,0,0,0,0,0,0,0,0,0]
    for AGV in AGV_classes_information:
        if AGV_classes_information[AGV].Status=='Dispatched':
            for b in list_departements:
                if b in AGV_classes_information[AGV].Current_location:
                   number_of_dispatched_vehicles_per_location[list_departements.index(b)]=number_of_dispatched_vehicles_per_location[list_departements.index(b)]+1
 
    
    number_of_currently_travelling_dispatched_vehicles_per_location=[0,0,0,0,0,0,0,0,0,0]
    for AGV in AGV_classes_information:
        if AGV_classes_information[AGV].Status=='Driving_to_dispatch_location':
            for b in list_departements:
                if b in AGV_classes_information[AGV].Dispatching_location:
                   number_of_currently_travelling_dispatched_vehicles_per_location[list_departements.index(b)]=number_of_currently_travelling_dispatched_vehicles_per_location[list_departements.index(b)]+1

    
    number_of_currently_Charging_vehicles_per_location=[0,0,0,0,0,0,0,0,0,0]
    for AGV in AGV_classes_information:
        if AGV_classes_information[AGV].Status=='Charging':
            for b in list_departements:
                if b in AGV_classes_information[AGV].Current_location:
                   number_of_currently_Charging_vehicles_per_location[list_departements.index(b)]=number_of_currently_Charging_vehicles_per_location[list_departements.index(b)]+1

    
    distance_to_every_departement_vector=Distance_to_every_departement_AGV(Location_AGV)
    
    demand_in_last_time_windows=[0,0,0,0,0,0,0,0,0,0]
    current_time_window=(Simulator.time-23300)//600
    for x in range(current_time_window-2, current_time_window+1):
        if x>=0:
            for z in range(0, len(list_departements)):
                demand_in_last_time_windows[z]=demand_in_last_time_windows[z]+Simulator.request_tracker[z,x]
            
            
    demand_future_windows_day_before= [0,0,0,0,0,0,0,0,0,0]
    current_time_window=(Simulator.time-23300)//600
    for x in range(current_time_window+1, current_time_window++4):
        if x<100:
            for z in range(0, len(list_departements)):
                demand_future_windows_day_before[z]=demand_future_windows_day_before[z]+Simulator.request_tracker[z,x]
           
        
    
    
    
    global_state=number_of_dispatched_vehicles_per_location+number_of_currently_travelling_dispatched_vehicles_per_location+number_of_currently_Charging_vehicles_per_location+[(Simulator.time-23300)]+[(Simulator.day%5)]+distance_to_every_departement_vector+demand_in_last_time_windows+demand_future_windows_day_before
   
    
    
    return(global_state)
                   
def Distance_to_every_departement_AGV(Location_AGV):
    Locations=[]
    for x in list_departements:
        z=x+'_U1'
        Locations.append(z)
    State_vector_distance=[]
    for Location in Locations:
        Time1=Time_calculator(Location_AGV, Location)
        Time2=Lift_move_time_calculator(Simulator.time, Location_AGV, Location)
        total_time=Time1+Time2
        State_vector_distance.append(total_time)
    
    return (State_vector_distance)
        
              
        
        
def Reward_calculator(Request, Time_picked_up):
    Reward=Request_dict_simulation[Request].Time_arrival-Time_picked_up
    Simulator.reward=Simulator.reward+Reward
    Simulator.score=Simulator.score+Reward
    
        
    
                    
            
class Dispatching_DQN:
    def __init__(self):
        self.last_state=[]
        self.new_state=[]
        self.action=[]
        self.reward=[]
        self.nr_steps=0
    
    def Add_step(self, AGV_location):
        self.new_state=np.array(State_vector_departement_maker(AGV_location))
        self.new_state=self.new_state.astype(np.float32)
        if self.nr_steps>=1:
            Agent.store_transition(self.last_state, self.action, float(Simulator.reward), self.new_state, done) 
        self.nr_steps+=1
        self.last_state=self.new_state
        self.action=Agent.choose_action(self.new_state)
        print(self.action)
        Agent.learn()



    
def Request_tracker(request_tracker_matrix, request_current_period,time) :   
    for request in request_current_period:
        for departement in list_departements:
            if departement in Request_dict_simulation[request].Pick_up_location:
                time_window=(time-23300)//600
                index_1=list_departements.index(departement)
                request_tracker_matrix[index_1, time_window]=request_tracker_matrix[index_1, time_window]+1


waiting_time_per_episode=[]
score_of_day=[]    
done=False    
class Simulator:
    def __init__(self):
        self.time=23300
        self.maxiterations=1
        self.day=0
        self.reward=0
        self.request_tracker=np.zeros((10,100))
        self.Simulation_iteration=0
        

    def Simulator(self):
        self.day=0
        self.time=23300 ##system starts at 6:30 A.M. calculated to seconds
        iteration=0
        request_without_an_AGV=[]
        self.score=0
        while self.maxiterations>self.Simulation_iteration:
            request_current_period=[]
            self.time=self.time+time_interval## first the time is updated
            
            request_current_period=Request_generator(self.day, self.time) ## the requests for that time_interval are calculated
            Request_tracker(self.request_tracker,request_current_period, self.time ) ## tracks all the requests to use in the state
            Request_total=request_without_an_AGV+request_current_period
            request_without_an_AGV=[]
            for request in Request_total: ## requests are assigned to an AGV
                if Assigning_request_to_AGVS(request)=='No AGV found':
                    request_without_an_AGV.append(request)
                
                     
                
            for AGV in AGV_classes_information:
                
                
                if AGV_classes_information[AGV].Status=='Idle': ## If AGV is idle and have requests assigned he will complete these requests
                    AGV_classes_information[AGV].Idle()
                
                if AGV_classes_information[AGV].Status=='Driving_to_request':
                    AGV_classes_information[AGV].Drive_to_request()               
                                          
                if AGV_classes_information[AGV].Status=='Driving_to_dispatch_location':
                   AGV_classes_information[AGV].Driving_to_dispatch_location()
                
                
                if AGV_classes_information[AGV].Status=='Transporting_request_dropoff':
                    AGV_classes_information[AGV].Transporting_request_dropoff()
                    
    
                    
                if AGV_classes_information[AGV].Status=='Driving_to_Charging':
                    AGV_classes_information[AGV].Driving_to_Charging()
                
                    
                if AGV_classes_information[AGV].Status=='Dispatched':
                    AGV_classes_information[AGV].Dispatched()
                
                if AGV_classes_information[AGV].Status=='Charging':
                    AGV_classes_information[AGV].Charging()
                    
            
            iteration+=1
            if (self.time-23300)%600==0: ## when a new time window arrives, request_tracker sets the request in that time_window to zero.
                for x in range(0,len(list_departements)):
                    z=(self.time-23300)//600
                    self.request_tracker[x,z]=0
                
            if self.time>80000:

                
                total_waiting_time=0
                if self.day>0:
                    start_range=when_days_end[self.day]
                else:
                    start_range=0
                end_range=when_days_end[self.day+1]
                total_waiting_time=0
                for x in range(start_range,end_range):
                    if x in Request_dict_simulation:
                        total_waiting_time+=(Request_dict_simulation[x].Time_picked_up-Request_dict_simulation[x].Time_arrival)
                
                average_waiting_time= total_waiting_time/(end_range-start_range)
                print('#####average waiting time of this episode is :' + str(average_waiting_time))
                
                score_of_day.append(self.score)
                self.score=0
                self.day+=1
                waiting_time_per_episode.append(average_waiting_time)
                if self.day==1:
                    self.day=0
                    self.Simulation_iteration+=1
                    with open("Waiting_time_5.txt", "w") as output:
                        output.write(str(waiting_time_per_episode))
                Reset_battery_when_day_over()
                self.time=23300
                done=True
                
                    

Day_slicer_input_requests()         
Simulator=Simulator()
Dispatching_DQN=Dispatching_DQN() 
Simulator.Simulator()


                    

            
        
                    
                
            
            
                         
                         
                        
                    
            
                    
                
            
                
            
                    
                    
                    
                    
                    
                    
        
                    
                    
                    
                    
                    
            
            
        
        
    
#    

        
        
    
        
        
        
        
    
                
        
                
    
#        