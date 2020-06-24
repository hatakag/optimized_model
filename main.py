import tensorflow as tf
import numpy
from include.Config import Config
from include.Model import build, training
from include.Test import get_hits
from include.Load import *

import warnings
warnings.filterwarnings("ignore")

'''
Follow the code style of GCN-Align:
https://github.com/1049451037/GCN-Align
'''

seed = 12306
np.random.seed(seed)
tf.set_random_seed(seed)

if __name__ == '__main__':
    e = len(set(loadfile(Config.e1, 1)) | set(loadfile(Config.e2, 1)))

    ILL = loadfile(Config.shuffled_ill, 2)
    illL = len(ILL)
    #np.random.shuffle(ILL) -- use this instead !shuf data/zh_en/ref_ent_ids -o data/zh_en/shuffled_ref_ent_ids
    train = np.array(ILL[:illL // 10 * Config.seed])
    test = ILL[illL // 10 * Config.seed:]
    
    ILL_r = loadfile(Config.ill_r, 2)
    ILL_r = np.array(ILL_r[:])

    KG1 = loadfile(Config.kg1, 3)
    KG2 = loadfile(Config.kg2, 3)

    output_prel_e, output_joint_e, output_r, loss_1, loss_2, loss_3, head, tail = build(
        Config.dim, Config.act_func, Config.alpha, Config.beta, Config.gamma, Config.k, Config.language[0:2], e, train, ILL_r, KG1 + KG2)
    vec, J = training(output_prel_e, output_joint_e, output_r, loss_1, loss_2, loss_3, 0.001, Config.epochs, train, e, Config.k, Config.s1, Config.s2, test, ILL_r, head, tail)
    print('loss:', J)
    print('Result:')
    get_hits(vec, test)
    
    numpy.save("output_" + Config.language, vec) 
    print("The output vector is saved in the file " + "output_" + Config.language + ".npy")
