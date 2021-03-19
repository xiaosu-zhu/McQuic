import torch
from torch import nn
from torch.distributions import Categorical, Bernoulli
from torch.nn import functional as F

def bernoulli_loglikelihood(b, logits):
    """Return logP of Bernoulli distribution

    Args:
        b ([type]): [N, D], 0 or 1 sample result
        logits ([type]): [N, D], logits.

    Returns:
        [type]: [description]
    """
    '''
    input: N*d; output: N*d
    '''
    return b * (-tf.nn.softplus(-logits)) + (1 - b) * (-logits - tf.nn.softplus(-logits))

def categorical_loglikelihood(b, logits):
    '''
    b is N*n_cv*n_class, one-hot vector in row
    logits is N*n_cv*n_class, softmax(logits) is prob
    return: N*n_cv
    '''
    lik_v = b*(logits-tf.reduce_logsumexp(logits,axis=-1,keep_dims=True))
    return tf.reduce_sum(lik_v,axis=-1)




class AugReinSwapMerge(nn.Module):
    def __init__(self):
        super().__init__()

        self._encoder = nn.Sequential(nn.Linear(encoderXD, 512), nn.LeakyReLU(negative_slope=0.2), nn.Linear(512, 256), nn.LeakyReLU(negative_slope=0.2), nn.Linear(256, encoderZD))

        self._decoder = nn.Sequential(nn.Flatten(), nn.Linear(encodeBD, 256), nn.LeakyReLU(negative_slope=0.2), nn.Linear(256, 512), nn.LeakyReLU(negative_slope=0.2), nn.Linear(512, encoderXD))


    def forward(self):
        pass



def encoder(x,z_dim):
    '''
    return logits [N,n_cv*(n_class-1)]
    z_dim is n_cv*(n_class-1)
    '''
    with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
        h = slim.stack(x, slim.fully_connected,[512,256],activation_fn=lrelu)
        z = tf.layers.dense(h, z_dim, name="encoder_out",activation = None)
    return z

def decoder(b,x_dim):
    '''
    return logits
    b is [N,n_cv,n_class]
    '''
    with tf.variable_scope("decoder",reuse=tf.AUTO_REUSE):
        b = Flatten(b)
        h = slim.stack(b, slim.fully_connected,[256,512],activation_fn=lrelu)
        logit_x = tf.layers.dense(h, x_dim, activation = None)
    return logit_x


def fun(x_binary, E, prior_logit0, z_concate):
    '''
    x_binary is N*d_x, E is N*n_cv*n_class, z_concate is N*n_cv*n_class
    prior_logit0 is n_cv*n_class
    calculate log p(x_star|E) + log p(E) - log q(E|x_star)
    return (N,)
    '''
    # [N, nCV, Class]
    logits_py = prior_logit0[None, ...].expand_as(E)
    # [N, Dx]
    # log p(x|z)
    logit_x = self._decoder(E)
    # [N, Dx] -> [N, ]
    # log p(x|z)
    log_p_x_given_z = Bernoulli(logits=logit_x).log_prob(x_binary).sum(-1)
    # [N, nCV] -> [N,]
    #log q(z|x)
    log_q_z_given_x = Categorical(logits=z_concate).log_prob(E).sum(1)
    #log p(z)
    log_p_z = Categorical(logits=logits_py).log_prob(E).sum(1)

    return - log_p_x_given_z - log_p_z + log_q_z_given_x

eps = 1e-10

def Fn(pai, prior_logit0, z_concate, x_star_u):
    '''
    pai is [N,n_cv,n_class]
    z_concate is [N,n_class]
    '''
    E = F.one_hot(((pai + eps).log() - z_concate[:, None, ...]).argmin(3), num_classes=n_class).float()
    return fun(x_star_u, E, prior_logit0, z_concate)

def get_loss(sess,data,total_batch):
    cost_eval = []
    for j in range(total_batch):
        xs,_ = data.next_batch(batch_size)
        cost_eval.append(sess.run(gen_loss0,{x:xs}))
    return np.mean(cost_eval)

def compt_F(sess, dirich, logits, xs):
    FF = np.zeros([batch_size, n_class, n_class])
    for i in range(n_class):
        for j in range(i,n_class):
            dirich_ij = np.copy(dirich)
            dirich_ij[:,:,[i,j]] = dirich_ij[:,:,[j,i]]
            s_ij  = to_categorical(np.argmin(np.log(dirich_ij+eps)-logits, axis = -1),num_classes=n_class)
            FF[:,i,j] = sess.run(F_ij,{x:xs, EE:s_ij})
            FF[:,j,i] = FF[:,i,j]
    return FF
