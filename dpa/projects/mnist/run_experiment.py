from dpa.evaluation import RepeatedAttack
from dpa import attacker
import numpy as np
import argparse
import os
import random
import tensorflow as tf
from tensorflow.keras.callbacks import LambdaCallback




if __name__ == '__main__':
    parser =argparse.ArgumentParser(description="DPA_Model_training")
    parser.add_argument("--rho", "-r", default=0.9, type=float)
    parser.add_argument("--local", "-ls", action="store_true")
    parser.add_argument("--delta", "-dlt", type=float, default=0.001)
    parser.add_argument("--worst_case", "-wc", action="store_true")
    parser.add_argument("--iterations", "-it", type=int, default=1)
    parser.add_argument("--epochs", "-ep", type=int, default=3)
    parser.add_argument("--sequential", "-sq", action="store_true")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.001)
    parser.add_argument("--l2_norm_clip", "-l2", type=float, default=3.0)
    parser.add_argument("--identifier", "-id", type=int,default=1)
    parser.add_argument("--bounded", "-bnd", action="store_true")
    parser.add_argument("--MNIST", "-mnst", action="store_true")

    args = parser.parse_args()
    print(vars(args))
    #for graph
    tf.compat.v1.disable_v2_behavior()
    #seed_value = 42
    #os.environ['TF_DETERMINISTIC_OPS'] = '1'
    #os.environ['PYTHONHASHSEED'] = str(seed_value)
    #random.seed(43)
    #np.random.seed(seed_value)
    #tf.random.set_seed(seed_value)

    learning_rate=args.learning_rate
    l2_norm_clip = args.l2_norm_clip
    sequential=args.sequential
    epochs=args.epochs
    iterations = args.iterations
    rho=args.rho
    delta = args.delta
    worst_case = args.worst_case
    local=args.local
    adversary = attacker.GaussianAttacker()
    epsilon = adversary.get_epsilon_for_confidence_bound(rho)
    run_id = args.identifier
    bound = args.bounded
    MNIST = args.MNIST

    repeated_attack = RepeatedAttack()
    results= repeated_attack.nn_attack_graph(epochs=epochs, 
                                        composed_delta=delta,
                                        rho=rho, 
                                        worst_case=worst_case, 
                                        local=local, 
                                        sequential=sequential, 
                                        iterations=iterations, 
                                        learning_rate = learning_rate, 
                                        l2_norm_clip=l2_norm_clip,
                                        bounded=bound,
                                        MNIST=MNIST)
    if MNIST:
        path = f"./experiments/rho_{rho}_delta_{delta}_local_{local}_bounded_{bound}_id_{run_id}.npz"#graph_rho_{rho}_delta_{delta}_local_{local}_bounded_{bound}_id_{run_id}.npz"
    else:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **results, args=vars(args))
    #os.system('sudo shutdown -h 0')




