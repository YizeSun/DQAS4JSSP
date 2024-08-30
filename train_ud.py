import argparse
import timeit
import yaml
import json
from circuit_ud import QDQN, Circuit_manager
from trainer import DQAS4RL
from multiprocessing import Pool


parser = argparse.ArgumentParser()

parser.add_argument("--num_layers", default=1, type=int)# *
parser.add_argument("--gamma", default=0.99, type=float)
parser.add_argument("--lr", default=0.1, type=float) # 0.01 for test of reducing lr
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--greedy", default=1., type=float)
parser.add_argument("--greedy_decay", default=0.99, type=int)
parser.add_argument("--greedy_min", default=0.01, type=float)
parser.add_argument("--update_model", default=1, type=int) # ensure that model updated for every epoch
parser.add_argument("--update_targ_model", default=50, type=int)
parser.add_argument("--memory_size", default=10000, type=int)
parser.add_argument("--loss_func", default='MSE', type=str)
parser.add_argument("--opt", default='Adam', type=str)
parser.add_argument("--epochs_train", default=1500, type=int)# *
parser.add_argument("--epochs_test", default=5, type=int)
parser.add_argument("--device", default='auto', type=str)
parser.add_argument("--early_stop", default=195, type=int)

# RL
parser.add_argument("--w_input", default=False, type=bool)
parser.add_argument("--w_output", default=False, type=bool)
parser.add_argument("--lr_input", default=0.001, type=float)
parser.add_argument("--lr_output", default=0.1, type=float)

parser.add_argument("--lr_struc", default=0.1, type=float) # 0.01
parser.add_argument("--max_steps", default=200, type=int) #TODO: fix for fl
parser.add_argument("--seed", default=1234, type=int)
parser.add_argument("--num_placeholders", default=10, type=int)# *
parser.add_argument("--opt_struc", default='Adam', type=str)
parser.add_argument("--structure_batch", default=10, type=int)
parser.add_argument("--num_qubits", default=10, type=int)# *
parser.add_argument("--struc_early_stop", default=1500, type=int) # *
parser.add_argument("--learning_step", default=10, type=int) # 9
parser.add_argument("--p_search", default=False, type=bool) # *
parser.add_argument("--p_search_lowerbound", default=2, type=int) # *
parser.add_argument("--p_search_period", default=0, type=int) # *

parser.add_argument("--data_reuploading", default=False, type=bool) #False
parser.add_argument("--use_sphc_struc", default=True, type=bool)
parser.add_argument("--barrier", default=False, type=bool)
parser.add_argument("--exp_name", default='cp', type=str)
parser.add_argument("--agent_task", default='default', type=str)
parser.add_argument("--noisy", default=False, type=bool)

parser.add_argument("--logging", default=True, type=bool)
parser.add_argument("--debug", default=False, type=bool)
parser.add_argument("--log_train_freq", default=1, type=int)
parser.add_argument("--log_eval_freq", default=20, type=int)
parser.add_argument("--log_ckp_freq", default=50, type=int)
parser.add_argument("--log_records_freq", default=50, type=int)

args = parser.parse_args()

def train(exp_name,ops):
    # print(f"exp name: {exp_name}, agent task: {agent_task}, agent name: {agent_name}")
    start = timeit.default_timer()

    sphc_struc = []
    # sphc_struc = ["CZ"]
    # sphc_struc = ["RY", "RZ", "CNOT"]
    sphc_ranges = [[*range(args.num_qubits)] for _ in range(len(sphc_struc))]

    cm = Circuit_manager(sphc_struc=sphc_struc
                        , sphc_ranges=sphc_ranges
                        , num_qubits=args.num_qubits
                        , num_placeholders=args.num_placeholders
                        , num_layers=args.num_layers
                        , ops=ops
                        , noisy=args.noisy
                        , learning_step=args.learning_step
                        )

    # Define quantum network
    qdqn = QDQN(cm=cm
            , w_input=args.w_input
            , w_output=args.w_output
            , data_reuploading=args.data_reuploading
            , barrier=args.barrier
            , seed=args.seed)

    #qdqn_target = QDQN(cm=cm
    #                , w_input=args.w_input
    #                , w_output=args.w_output
    #                , data_reuploading=args.data_reuploading
    #                , barrier=args.barrier
    #                , seed=args.seed)

    dqas4rl = DQAS4RL(qdqn=qdqn,
                      gamma=args.gamma,
                      lr=args.lr,
                      lr_struc=args.lr_struc,
                      batch_size=args.batch_size,
                      update_model=args.update_model,
                      memory_size=args.memory_size,
                      max_steps=args.max_steps,
                      seed=args.seed,
                      cm=cm,
                      prob_max=0,
                      lr_in=args.lr_input,
                      lr_out=args.lr_output,
                      loss_func=args.loss_func,
                      opt=args.opt,
                      opt_struc=args.opt_struc,
                      device=args.device,
                      logging=args.logging,
                      verbose=False,
                      early_stop=args.early_stop,
                      structure_batch=args.structure_batch,
                      debug=args.debug,
                      exp_name=exp_name,
                      struc_learning=cm.learning_state,
                      total_epochs=args.epochs_train,
                      p_search=args.p_search,
                      p_search_lowerbound=args.p_search_lowerbound,
                      p_search_period=args.p_search_period,
                      struc_early_stop=args.struc_early_stop)

    if args.logging:
        with open(dqas4rl.log_dir + 'config.yaml', 'w') as f:
            yaml.safe_dump(args.__dict__, f, indent=2)

    dqas4rl.learn(num_eval_epochs=args.epochs_test,
                  log_train_freq=args.log_train_freq,
                  log_eval_freq=args.log_eval_freq,
                  log_ckp_freq=args.log_ckp_freq,
                  log_records_freq=args.log_records_freq)

    stop = timeit.default_timer()
    with open(dqas4rl.log_dir + 'total_time.json', 'w') as f:
        json.dump(f'total time cost: {stop-start}', f,  indent=4)
    print(f'total time cost: {stop-start}')

def main():
    ops1 = {0: ("RY", [0, 1, 2, 3, 4, 5, 6])
        , 1: ("RY", [1, 2, 3, 4, 5, 6, 7])
        , 2: ("RY", [2, 3, 4, 5, 6, 7, 8])
        , 3: ("RY", [3, 4, 5, 6, 7, 8, 9])
        , 4: ("RY", [0, 1, 2, 3, 4, 5])
        , 5: ("RY", [1, 2, 3, 4, 5, 6])
        , 6: ("RY", [2, 3, 4, 5, 6, 7])
        , 7: ("RY", [3, 4, 5, 6, 7, 8])
        , 8: ("RY", [4, 5, 6, 7, 8, 9])
        , 9: ("RY", [0, 1, 2, 3, 4])
        , 10: ("RY", [1, 2, 3, 4, 5])
        , 11: ("RY", [2, 3, 4, 5, 6])
        , 12: ("RY", [3, 4, 5, 6, 7])
        , 13: ("RY", [4, 5, 6, 7, 8])
        , 14: ("RY", [5, 6, 7, 8, 9])
        , 15: ("RY", [0, 1, 2, 3])
        , 16: ("RY", [1, 2, 3, 4])
        , 17: ("RY", [2, 3, 4, 5])
        , 18: ("RY", [3, 4, 5, 6])
        , 19: ("RY", [4, 5, 6, 7])
        , 20: ("RY", [5, 6, 7, 8])
        , 21: ("RY", [6, 7, 8, 9])
        , 22: ("RY", [0, 1, 2])
        , 23: ("RY", [1, 2, 3])
        , 24: ("RY", [2, 3, 4])
        , 25: ("RY", [3, 4, 5])
        , 26: ("RY", [4, 5, 6])
        , 27: ("RY", [5, 6, 7])
        , 28: ("RY", [6, 7, 8])
        , 29: ("RY", [7, 8, 9])
        , 30: ("RY", [0, 1])
        , 31: ("RY", [1, 2])
        , 32: ("RY", [2, 3])
        , 33: ("RY", [3, 4])
        , 34: ("RY", [4, 5])
        , 35: ("RY", [5, 6])
        , 36: ("RY", [6, 7])
        , 37: ("RY", [7, 8])
        , 38: ("RY", [8, 9])
        , 39: ("RZ", [0, 1, 2, 3, 4, 5, 6])
        , 40: ("RZ", [1, 2, 3, 4, 5, 6, 7])
        , 41: ("RZ", [2, 3, 4, 5, 6, 7, 8])
        , 42: ("RZ", [3, 4, 5, 6, 7, 8, 9])
        , 43: ("RZ", [0, 1, 2, 3, 4, 5])
        , 44: ("RZ", [1, 2, 3, 4, 5, 6])
        , 45: ("RZ", [2, 3, 4, 5, 6, 7])
        , 46: ("RZ", [3, 4, 5, 6, 7, 8])
        , 47: ("RZ", [4, 5, 6, 7, 8, 9])
        , 48: ("RZ", [0, 1, 2, 3, 4])
        , 49: ("RZ", [1, 2, 3, 4, 5])
        , 50: ("RZ", [2, 3, 4, 5, 6])
        , 51: ("RZ", [3, 4, 5, 6, 7])
        , 52: ("RZ", [4, 5, 6, 7, 8])
        , 53: ("RZ", [5, 6, 7, 8, 9])
        , 54: ("RZ", [0, 1, 2, 3])
        , 55: ("RZ", [1, 2, 3, 4])
        , 56: ("RZ", [2, 3, 4, 5])
        , 57: ("RZ", [3, 4, 5, 6])
        , 58: ("RZ", [4, 5, 6, 7])
        , 59: ("RZ", [5, 6, 7, 8])
        , 60: ("RZ", [6, 7, 8, 9])
        , 61: ("RZ", [0, 1, 2])
        , 62: ("RZ", [1, 2, 3])
        , 63: ("RZ", [2, 3, 4])
        , 64: ("RZ", [3, 4, 5])
        , 65: ("RZ", [4, 5, 6])
        , 66: ("RZ", [5, 6, 7])
        , 67: ("RZ", [6, 7, 8])
        , 68: ("RZ", [7, 8, 9])
        , 69: ("RZ", [0, 1])
        , 70: ("RZ", [1, 2])
        , 71: ("RZ", [2, 3])
        , 72: ("RZ", [3, 4])
        , 73: ("RZ", [4, 5])
        , 74: ("RZ", [5, 6])
        , 75: ("RZ", [6, 7])
        , 76: ("RZ", [7, 8])
        , 77: ("RZ", [8, 9])
        , 78: ("CZ", [0, 1, 2, 3, 4, 5, 6, 7, 8])
        , 79: ("CNOT", [0, 1, 2, 3, 4, 5, 6, 7, 8])
        , 80: ("E", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}

    ops2 = {0: ("RY", [0, 1, 2, 3, 4, 5])
        , 1: ("RY", [1, 2, 3, 4, 5, 6])
        , 2: ("RY", [2, 3, 4, 5, 6, 7])
        , 3: ("RY", [3, 4, 5, 6, 7, 8])
        , 4: ("RY", [4, 5, 6, 7, 8, 9])
        , 5: ("RY", [0, 1, 2, 3, 4])
        , 6: ("RY", [1, 2, 3, 4, 5])
        , 7: ("RY", [2, 3, 4, 5, 6])
        , 8: ("RY", [3, 4, 5, 6, 7])
        , 9: ("RY", [4, 5, 6, 7, 8])
        , 10: ("RY", [5, 6, 7, 8, 9])
        , 11: ("RY", [0, 1, 2, 3])
        , 12: ("RY", [1, 2, 3, 4])
        , 13: ("RY", [2, 3, 4, 5])
        , 14: ("RY", [3, 4, 5, 6])
        , 15: ("RY", [4, 5, 6, 7])
        , 16: ("RY", [5, 6, 7, 8])
        , 17: ("RY", [6, 7, 8, 9])
        , 18: ("RY", [0, 1, 2])
        , 19: ("RY", [1, 2, 3])
        , 20: ("RY", [2, 3, 4])
        , 21: ("RY", [3, 4, 5])
        , 22: ("RY", [4, 5, 6])
        , 23: ("RY", [5, 6, 7])
        , 24: ("RY", [6, 7, 8])
        , 25: ("RY", [7, 8, 9])
        , 26: ("RY", [0, 1])
        , 27: ("RY", [1, 2])
        , 28: ("RY", [2, 3])
        , 29: ("RY", [3, 4])
        , 30: ("RY", [4, 5])
        , 31: ("RY", [5, 6])
        , 32: ("RY", [6, 7])
        , 33: ("RY", [7, 8])
        , 34: ("RY", [8, 9])
        , 35: ("RZ", [0, 1, 2, 3, 4, 5])
        , 36: ("RZ", [1, 2, 3, 4, 5, 6])
        , 37: ("RZ", [2, 3, 4, 5, 6, 7])
        , 38: ("RZ", [3, 4, 5, 6, 7, 8])
        , 39: ("RZ", [4, 5, 6, 7, 8, 9])
        , 40: ("RZ", [0, 1, 2, 3, 4])
        , 41: ("RZ", [1, 2, 3, 4, 5])
        , 42: ("RZ", [2, 3, 4, 5, 6])
        , 43: ("RZ", [3, 4, 5, 6, 7])
        , 44: ("RZ", [4, 5, 6, 7, 8])
        , 45: ("RZ", [5, 6, 7, 8, 9])
        , 46: ("RZ", [0, 1, 2, 3])
        , 47: ("RZ", [1, 2, 3, 4])
        , 48: ("RZ", [2, 3, 4, 5])
        , 49: ("RZ", [3, 4, 5, 6])
        , 50: ("RZ", [4, 5, 6, 7])
        , 51: ("RZ", [5, 6, 7, 8])
        , 52: ("RZ", [6, 7, 8, 9])
        , 53: ("RZ", [0, 1, 2])
        , 54: ("RZ", [1, 2, 3])
        , 55: ("RZ", [2, 3, 4])
        , 56: ("RZ", [3, 4, 5])
        , 57: ("RZ", [4, 5, 6])
        , 58: ("RZ", [5, 6, 7])
        , 59: ("RZ", [6, 7, 8])
        , 60: ("RZ", [7, 8, 9])
        , 61: ("RZ", [0, 1])
        , 62: ("RZ", [1, 2])
        , 63: ("RZ", [2, 3])
        , 64: ("RZ", [3, 4])
        , 65: ("RZ", [4, 5])
        , 66: ("RZ", [5, 6])
        , 67: ("RZ", [6, 7])
        , 68: ("RZ", [7, 8])
        , 69: ("RZ", [8, 9])
        , 70: ("CZ", [0, 1, 2, 3, 4, 5, 6, 7, 8])
        , 71: ("CNOT", [0, 1, 2, 3, 4, 5, 6, 7, 8])
        , 72: ("E", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}

    ops3 = {0: ("RY", [0, 1, 2, 3, 4])
        , 1: ("RY", [1, 2, 3, 4, 5])
        , 2: ("RY", [2, 3, 4, 5, 6])
        , 3: ("RY", [3, 4, 5, 6, 7])
        , 4: ("RY", [4, 5, 6, 7, 8])
        , 5: ("RY", [5, 6, 7, 8, 9])
        , 6: ("RY", [0, 1, 2, 3])
        , 7: ("RY", [1, 2, 3, 4])
        , 8: ("RY", [2, 3, 4, 5])
        , 9: ("RY", [3, 4, 5, 6])
        , 10: ("RY", [4, 5, 6, 7])
        , 11: ("RY", [5, 6, 7, 8])
        , 12: ("RY", [6, 7, 8, 9])
        , 13: ("RY", [0, 1, 2])
        , 14: ("RY", [1, 2, 3])
        , 15: ("RY", [2, 3, 4])
        , 16: ("RY", [3, 4, 5])
        , 17: ("RY", [4, 5, 6])
        , 18: ("RY", [5, 6, 7])
        , 19: ("RY", [6, 7, 8])
        , 20: ("RY", [7, 8, 9])
        , 21: ("RY", [0, 1])
        , 22: ("RY", [1, 2])
        , 23: ("RY", [2, 3])
        , 24: ("RY", [3, 4])
        , 25: ("RY", [4, 5])
        , 26: ("RY", [5, 6])
        , 27: ("RY", [6, 7])
        , 28: ("RY", [7, 8])
        , 29: ("RY", [8, 9])
        , 30: ("RZ", [0, 1, 2, 3, 4])
        , 31: ("RZ", [1, 2, 3, 4, 5])
        , 32: ("RZ", [2, 3, 4, 5, 6])
        , 33: ("RZ", [3, 4, 5, 6, 7])
        , 34: ("RZ", [4, 5, 6, 7, 8])
        , 35: ("RZ", [5, 6, 7, 8, 9])
        , 36: ("RZ", [0, 1, 2, 3])
        , 37: ("RZ", [1, 2, 3, 4])
        , 38: ("RZ", [2, 3, 4, 5])
        , 39: ("RZ", [3, 4, 5, 6])
        , 40: ("RZ", [4, 5, 6, 7])
        , 41: ("RZ", [5, 6, 7, 8])
        , 42: ("RZ", [6, 7, 8, 9])
        , 43: ("RZ", [0, 1, 2])
        , 44: ("RZ", [1, 2, 3])
        , 45: ("RZ", [2, 3, 4])
        , 46: ("RZ", [3, 4, 5])
        , 47: ("RZ", [4, 5, 6])
        , 48: ("RZ", [5, 6, 7])
        , 49: ("RZ", [6, 7, 8])
        , 50: ("RZ", [7, 8, 9])
        , 51: ("RZ", [0, 1])
        , 52: ("RZ", [1, 2])
        , 53: ("RZ", [2, 3])
        , 54: ("RZ", [3, 4])
        , 55: ("RZ", [4, 5])
        , 56: ("RZ", [5, 6])
        , 57: ("RZ", [6, 7])
        , 58: ("RZ", [7, 8])
        , 59: ("RZ", [8, 9])
        , 60: ("CZ", [0, 1, 2, 3, 4, 5, 6, 7, 8])
        , 61: ("CNOT", [0, 1, 2, 3, 4, 5, 6, 7, 8])
        , 62: ("E", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}

    ops4 = {0: ("RY", [0, 1, 2, 3])
        , 1: ("RY", [1, 2, 3, 4])
        , 2: ("RY", [2, 3, 4, 5])
        , 3: ("RY", [3, 4, 5, 6])
        , 4: ("RY", [4, 5, 6, 7])
        , 5: ("RY", [5, 6, 7, 8])
        , 6: ("RY", [6, 7, 8, 9])
        , 7: ("RY", [0, 1, 2])
        , 8: ("RY", [1, 2, 3])
        , 9: ("RY", [2, 3, 4])
        , 10: ("RY", [3, 4, 5])
        , 11: ("RY", [4, 5, 6])
        , 12: ("RY", [5, 6, 7])
        , 13: ("RY", [6, 7, 8])
        , 14: ("RY", [7, 8, 9])
        , 15: ("RY", [0, 1])
        , 16: ("RY", [1, 2])
        , 17: ("RY", [2, 3])
        , 18: ("RY", [3, 4])
        , 19: ("RY", [4, 5])
        , 20: ("RY", [5, 6])
        , 21: ("RY", [6, 7])
        , 22: ("RY", [7, 8])
        , 23: ("RY", [8, 9])
        , 24: ("RZ", [0, 1, 2, 3])
        , 25: ("RZ", [1, 2, 3, 4])
        , 26: ("RZ", [2, 3, 4, 5])
        , 27: ("RZ", [3, 4, 5, 6])
        , 28: ("RZ", [4, 5, 6, 7])
        , 29: ("RZ", [5, 6, 7, 8])
        , 30: ("RZ", [6, 7, 8, 9])
        , 31: ("RZ", [0, 1, 2])
        , 32: ("RZ", [1, 2, 3])
        , 33: ("RZ", [2, 3, 4])
        , 34: ("RZ", [3, 4, 5])
        , 35: ("RZ", [4, 5, 6])
        , 36: ("RZ", [5, 6, 7])
        , 37: ("RZ", [6, 7, 8])
        , 38: ("RZ", [7, 8, 9])
        , 39: ("RZ", [0, 1])
        , 40: ("RZ", [1, 2])
        , 41: ("RZ", [2, 3])
        , 42: ("RZ", [3, 4])
        , 43: ("RZ", [4, 5])
        , 44: ("RZ", [5, 6])
        , 45: ("RZ", [6, 7])
        , 46: ("RZ", [7, 8])
        , 47: ("RZ", [8, 9])
        , 48: ("CZ", [0, 1, 2, 3, 4, 5, 6, 7, 8])
        , 49: ("CNOT", [0, 1, 2, 3, 4, 5, 6, 7, 8])
        , 50: ("E", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}

    ops5 = {0: ("RY", [0, 1, 2])
        , 1: ("RY", [1, 2, 3])
        , 2: ("RY", [2, 3, 4])
        , 3: ("RY", [3, 4, 5])
        , 4: ("RY", [4, 5, 6])
        , 5: ("RY", [5, 6, 7])
        , 6: ("RY", [6, 7, 8])
        , 7: ("RY", [7, 8, 9])
        , 8: ("RY", [0, 1])
        , 9: ("RY", [1, 2])
        , 10: ("RY", [2, 3])
        , 11: ("RY", [3, 4])
        , 12: ("RY", [4, 5])
        , 13: ("RY", [5, 6])
        , 14: ("RY", [6, 7])
        , 15: ("RY", [7, 8])
        , 16: ("RY", [8, 9])
        , 17: ("RZ", [0, 1, 2])
        , 18: ("RZ", [1, 2, 3])
        , 19: ("RZ", [2, 3, 4])
        , 20: ("RZ", [3, 4, 5])
        , 21: ("RZ", [4, 5, 6])
        , 22: ("RZ", [5, 6, 7])
        , 23: ("RZ", [6, 7, 8])
        , 24: ("RZ", [7, 8, 9])
        , 25: ("RZ", [0, 1])
        , 26: ("RZ", [1, 2])
        , 27: ("RZ", [2, 3])
        , 28: ("RZ", [3, 4])
        , 29: ("RZ", [4, 5])
        , 30: ("RZ", [5, 6])
        , 31: ("RZ", [6, 7])
        , 32: ("RZ", [7, 8])
        , 33: ("RZ", [8, 9])
        , 34: ("CZ", [0, 1, 2, 3, 4, 5, 6, 7, 8])
        , 35: ("CNOT", [0, 1, 2, 3, 4, 5, 6, 7, 8])
        , 36: ("E", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}

    ops6 = {0: ("RY", [0, 1])
        , 1: ("RY", [1, 2])
        , 2: ("RY", [2, 3])
        , 3: ("RY", [3, 4])
        , 4: ("RY", [4, 5])
        , 5: ("RY", [5, 6])
        , 6: ("RY", [6, 7])
        , 7: ("RY", [7, 8])
        , 8: ("RY", [8, 9])
        , 9: ("RZ", [0, 1])
        , 10: ("RZ", [1, 2])
        , 11: ("RZ", [2, 3])
        , 12: ("RZ", [3, 4])
        , 13: ("RZ", [4, 5])
        , 14: ("RZ", [5, 6])
        , 15: ("RZ", [6, 7])
        , 16: ("RZ", [7, 8])
        , 17: ("RZ", [8, 9])
        , 18: ("CZ", [0, 1, 2, 3, 4, 5, 6, 7, 8])
        , 19: ("CNOT", [0, 1, 2, 3, 4, 5, 6, 7, 8])
        , 20: ("E", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}

    ops7 = {0: ("RY", [0, 1, 2, 3, 4, 5])
        , 1: ("RY", [1, 2, 3, 4, 5, 6])
        , 2: ("RY", [2, 3, 4, 5, 6, 7])
        , 3: ("RY", [3, 4, 5, 6, 7, 8])
        , 4: ("RY", [4, 5, 6, 7, 8, 9])
        , 5: ("RZ", [0, 1, 2, 3, 4, 5])
        , 6: ("RZ", [1, 2, 3, 4, 5, 6])
        , 7: ("RZ", [2, 3, 4, 5, 6, 7])
        , 8: ("RZ", [3, 4, 5, 6, 7, 8])
        , 9: ("RZ", [4, 5, 6, 7, 8, 9])
        , 10: ("CZ", [0, 1, 2, 3, 4, 5, 6, 7, 8])
        , 11: ("CNOT", [0, 1, 2, 3, 4, 5, 6, 7, 8])
        , 12: ("E", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}

    ops8 = {0: ("RY", [0, 1, 2, 3, 4])
        , 1: ("RY", [1, 2, 3, 4, 5])
        , 2: ("RY", [2, 3, 4, 5, 6])
        , 3: ("RY", [3, 4, 5, 6, 7])
        , 4: ("RY", [4, 5, 6, 7, 8])
        , 5: ("RZ", [0, 1, 2, 3, 4])
        , 6: ("RZ", [1, 2, 3, 4, 5])
        , 7: ("RZ", [2, 3, 4, 5, 6])
        , 8: ("RZ", [3, 4, 5, 6, 7])
        , 9: ("RZ", [4, 5, 6, 7, 8])
        , 10: ("RZ", [5, 6, 7, 8, 9])
        , 11: ("CZ", [0, 1, 2, 3, 4, 5, 6, 7, 8])
        , 12: ("CNOT", [0, 1, 2, 3, 4, 5, 6, 7, 8])
        , 13: ("E", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}

    ops9 = {0: ("RY", [0, 1, 2, 3])
        , 1: ("RY", [1, 2, 3, 4])
        , 2: ("RY", [2, 3, 4, 5])
        , 3: ("RY", [3, 4, 5, 6])
        , 4: ("RY", [4, 5, 6, 7])
        , 5: ("RY", [5, 6, 7, 8])
        , 6: ("RY", [6, 7, 8, 9])
        , 7: ("RZ", [0, 1, 2, 3])
        , 8: ("RZ", [1, 2, 3, 4])
        , 9: ("RZ", [2, 3, 4, 5])
        , 10: ("RZ", [3, 4, 5, 6])
        , 11: ("RZ", [4, 5, 6, 7])
        , 12: ("RZ", [5, 6, 7, 8])
        , 13: ("RZ", [6, 7, 8, 9])
        , 14: ("CZ", [0, 1, 2, 3, 4, 5, 6, 7, 8])
        , 15: ("CNOT", [0, 1, 2, 3, 4, 5, 6, 7, 8])
        , 16: ("E", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}

    ops10 = {0: ("RY", [0, 1, 2])
        , 1: ("RY", [1, 2, 3])
        , 2: ("RY", [2, 3, 4])
        , 3: ("RY", [3, 4, 5])
        , 4: ("RY", [4, 5, 6])
        , 5: ("RY", [5, 6, 7])
        , 6: ("RY", [6, 7, 8])
        , 7: ("RY", [7, 8, 9])
        , 8: ("RZ", [0, 1, 2])
        , 9: ("RZ", [1, 2, 3])
        , 10: ("RZ", [2, 3, 4])
        , 11: ("RZ", [3, 4, 5])
        , 12: ("RZ", [4, 5, 6])
        , 13: ("RZ", [5, 6, 7])
        , 14: ("RZ", [6, 7, 8])
        , 15: ("RZ", [7, 8, 9])
        , 16: ("CZ", [0, 1, 2, 3, 4, 5, 6, 7, 8])
        , 17: ("CNOT", [0, 1, 2, 3, 4, 5, 6, 7, 8])
        , 18: ("E", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}


    ops1 = [ops1, ops2, ops3, ops4, ops5]
    ops2 = [ops6, ops7, ops8, ops9, ops10]

    name1 = 'op10-9-01-l1-10-10-0.05'
    name2 = 'op10-9-02-l1-10-10-0.05'
    name3 = 'op10-9-03-l1-10-10-0.05'
    name4 = 'op10-9-04-l1-10-10-0.05'
    name5 = 'op10-9-05-l1-10-10-0.05'
    name6 = 'op10-9-06-l1-10-10-0.05'
    name7 = 'op10-9-07-l1-10-10-0.05'
    name8 = 'op10-9-08-l1-10-10-0.05'
    name9 = 'op10-9-09-l1-10-10-0.05'
    name10 = 'op10-9-10-l1-10-10-0.05'


    exp_name1 = [name1, name2, name3, name4, name5]
    exp_name2 = [name6, name7, name8, name9, name10]

    #lrs = [0.02, 0.05, 0.1]

    #earlys = [1]

    with Pool() as pool:
        pool.starmap(train, [(name, op) for name, op in zip(exp_name2, ops2)])

if __name__ == '__main__':
    main()