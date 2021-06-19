import numpy as np

obs_states_map = {'H': 0, 'T': 1}


def Forward(A, B, Pi, O):
    '''
    Args:
        A: (N, N) transition probability matrix
        B: (N, M) emission probability matrix
        Pi: (N,) initial state distribution
        O: a list of observations

    Returns: P(O|A, B, Pi)

    '''
    O = [obs_states_map[k] for k in O]
    A = np.asarray(A)
    B = np.asarray(B)
    O = np.asarray(O)
    Pi = np.asarray(Pi)

    alpha = Pi * B[:, O[0]]
    for o in O[1:]:
        condition_prob_obs = B[:, o]
        condition_prob_obs = condition_prob_obs[:, None]
        alpha = (condition_prob_obs * A.T) @ alpha

    return np.sum(alpha)

def Backward(A, B, Pi, O):
    '''
    Args:
        A: (N, N) transition probability matrix
        B: (N, M) emission probability matrix
        Pi: (N,) initial state distribution
        O: a list of observations

    Returns: P(O|A, B, Pi)

    '''
    O = [obs_states_map[k] for k in O]
    A = np.asarray(A)
    B = np.asarray(B)
    O = np.asarray(O)
    Pi = np.asarray(Pi)

    beta = np.ones(len(A))
    beta_list = [beta]
    for o in O[:0:-1]:
        condition_prob_obs = B[:, o]
        condition_prob_obs = condition_prob_obs[None, :]
        beta = (condition_prob_obs * A) @ beta
        beta_list.append(beta)
    beta_list = beta_list[::-1]
    res = np.sum(beta_list[0] * Pi * B[:, O[0]])

    return res


def Viterbi(A, B, Pi, O):
    '''
    Args:
        A: (N, N) transition probability matrix
        B: (N, M) emission probability matrix
        Pi: (N,) initial state distribution
        O: a list of observations

    Returns: P(O|A, B, Pi)

    '''
    O = [obs_states_map[k] for k in O]
    A = np.asarray(A)
    B = np.asarray(B)
    O = np.asarray(O)
    Pi = np.asarray(Pi)

    N, M = B.shape[0], len(O)
    delta = np.zeros((M, N))
    phi = np.zeros((M, N), dtype=np.int)
    delta[0] = Pi * B[:,O[0]]
    for i in range(1, M):
        obs = O[i]
        prob = (delta[i-1][:, None] * A)
        phi[i] = np.argmax(prob, axis=0)
        delta[i] = np.max(prob, axis=0) * B[:, obs]
    vit_score = np.max(delta[-1])
    path = [np.argmax(delta[-1])]
    for i in range(M-1, 0, -1):
        path.append(phi[i, path[-1]])
    return vit_score, path[::-1]


def BaumWelch(N, A, B, Pi, T):
    '''
    The implemented BaumWelch only support a single list of observations, for
    multiple sets of observations, see BaumWelchMultiObs
    Args:
        N: iteration number
        A: (N, N) transition probability matrix
        B: (N, M) emission probability matrix
        Pi: (N,) initial state distribution
        T: a list of observation, such as ['H', 'T', 'H']

    Returns:
        re-estimated A, B, Pi

    '''
    O = [obs_states_map[k] for k in T]
    A = np.asarray(A).copy()
    B = np.asarray(B).copy()
    O = np.asarray(O).copy()
    Pi = np.asarray(Pi).copy()

    for iteration in range(N):
        # alpha
        alpha = Pi * B[:, O[0]]
        alpha_list = [alpha]
        for o in O[1:]:
            condition_prob_obs = B[:, o]
            condition_prob_obs = condition_prob_obs[:, None]
            alpha = (condition_prob_obs * A.T) @ alpha
            alpha_list.append(alpha)

        # beta
        beta = np.ones(len(A))
        beta_list = [beta]
        for o in O[:0:-1]:
            condition_prob_obs = B[:, o]
            condition_prob_obs = condition_prob_obs[None, :]
            beta = (condition_prob_obs * A) @ beta
            beta_list.append(beta)
        beta_list = beta_list[::-1]
        alpha_matrix = np.asarray(alpha_list)   # T x N
        beta_matrix = np.asarray(beta_list)     # T x N


        gamma = alpha_matrix * beta_matrix # T x N
        gamma = gamma / np.sum(gamma, axis = 1, keepdims = True)
        beta_temp = beta_matrix * B.T[O]
        xi = alpha_matrix[:-1, :, None] * A[None, ...] * beta_temp[1:, None, :]
        xi = xi / np.sum(xi, axis=(1, 2), keepdims=True)
        A = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0)[:, None]

        gamma_sum = np.sum(gamma, axis=0)
        for i in range(B.shape[1]):
            mask = O == i
            B[:, i] = np.sum(gamma[mask], axis=0) / gamma_sum

        Pi = gamma[0]
    return A, B, Pi


def BaumWelchMultiSeq(N, A, B, Pi, T):
    '''
    seems as BaumWelch, but support multiple sets of observations
    Args:
        N: iteration number
        A: (N, N) transition probability matrix
        B: (N, M) emission probability matrix
        Pi: (N,) initial state distribution
        T: a list of observations, such as [['H', 'T', 'H'],['H', 'T']]

    Returns:
        re-estimated A, B, Pi

    '''
    T = [[obs_states_map[k] for k in O] for O in T]
    A = np.asarray(A).copy()
    B = np.asarray(B).copy()
    T = [np.asarray(O).copy() for O in T]
    Pi = np.asarray(Pi).copy()

    for iteration in range(N):
        numerators = [np.zeros_like(A), np.zeros_like(B), np.zeros_like(Pi)]
        denominators = [np.zeros((A.shape[0], 1)), np.zeros_like(B), len(T)]
        for O in T:

            # alpha
            alpha = Pi * B[:, O[0]]
            alpha_list = [alpha]
            for o in O[1:]:
                condition_prob_obs = B[:, o]
                condition_prob_obs = condition_prob_obs[:, None]
                alpha = (condition_prob_obs * A.T) @ alpha
                alpha_list.append(alpha)

            # beta
            beta = np.ones(len(A))
            beta_list = [beta]
            for o in O[:0:-1]:
                condition_prob_obs = B[:, o]
                condition_prob_obs = condition_prob_obs[None, :]
                beta = (condition_prob_obs * A) @ beta
                beta_list.append(beta)
            beta_list = beta_list[::-1]
            alpha_matrix = np.asarray(alpha_list)   # T x N
            beta_matrix = np.asarray(beta_list)     # T x N


            gamma = alpha_matrix * beta_matrix # T x N
            gamma = gamma / np.sum(gamma, axis = 1, keepdims = True)
            beta_temp = beta_matrix * B.T[O]
            xi = alpha_matrix[:-1, :, None] * A[None, ...] * beta_temp[1:, None, :]
            xi = xi / np.sum(xi, axis=(1, 2), keepdims=True)
            numerators[0] += np.sum(xi, axis=0)
            denominators[0] += np.sum(gamma[:-1], axis=0)[:, None]

            gamma_sum = np.sum(gamma, axis=0)
            for i in range(B.shape[1]):
                mask = O == i
                numerators[1][:, i] += np.sum(gamma[mask], axis=0)
                denominators[1][:, i] += gamma_sum

            numerators[2] += gamma[0]

        A, B, Pi = (n/d for n,d in zip(numerators, denominators))
    return A, B, Pi


if __name__ == '__main__':
    N = 100
    A = [[0.2, 0.8],    # A[p,q] = P(i_q|i_p)
         [0.6, 0.4]]
    B = [[0.7, 0.3],    # B[p,q] = P(o_q|i_p)
         [0.4, 0.6]]
    Pi = [1.0, 0.0]     # init hidden state probability
    # There are two observation states H and T: H --> 0, T -->1
    O = ['H', 'T', 'H', 'H', 'T', 'H', 'T'] # observations
    print(Forward(A, B, Pi, O))
    print(Backward(A, B, Pi, O))
    print(Viterbi(A, B, Pi, O)) # note that init state start from 0

    BaumWelch(N, A, B, Pi, O)
    BaumWelchMultiSeq(N, A, B, Pi, [['H', 'T', 'T'], ['H', 'H']])

