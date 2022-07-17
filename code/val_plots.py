import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

DAYS_IN_AWARD_PERIOD = 28
AWARD_PERIODS_PER_YEAR = 365.25 / DAYS_IN_AWARD_PERIOD
SLOTS_PER_AWARD_PERIOD = DAYS_IN_AWARD_PERIOD * 24 * 60 * 60 / 12
NET_VALIDATORS = 405000  #number of validators from https://beaconcha.in/
FLASHBOTS_DATA = np.genfromtxt('blockReward.csv', delimiter=',')


def calc_ppv_ls(n_validators, n_years, n_trials, n_smoothie=None, per_validator=False):
    # treat solitarius as a case where we are all the validators; then the rest can be
    # handled the same for both solitarius and smoothifiers
    if n_smoothie is None:
        n_smoothie = n_validators

    # get expected number of proposals
    n_periods = n_years * AWARD_PERIODS_PER_YEAR
    net_slots = int(round(
        SLOTS_PER_AWARD_PERIOD * n_smoothie * n_periods))  # TODO -- too big - hitting i64

    max_i32 = 2**30 - 1
    if net_slots > max_i32:  # scipy.stats uses int32s in it, so need to keep it to an ok size
        slots = max_i32
        proposal_scalar = int(round(net_slots / slots))
    else:
        slots = net_slots
        proposal_scalar = 1

    proposals_ls = scipy.stats.binom.rvs(slots, (1 / NET_VALIDATORS), loc=0, size=n_trials)
    proposals_ls = [proposals * proposal_scalar for proposals in proposals_ls]

    total_rewards_ls = []
    for proposals in proposals_ls:
        block_reward_arr = np.random.choice(FLASHBOTS_DATA, proposals, replace=True)
        total_rewards_ls.append(np.sum(block_reward_arr) * 1e18)  # convert wei to ETH

    if per_validator:
        reward_scalar = 1 / n_smoothie
    else:
        reward_scalar = n_validators / n_smoothie
    total_rewards_ls = [reward * reward_scalar for reward in total_rewards_ls]

    return total_rewards_ls


if __name__ == '__main__':
    # print(calc_ppv_ls(n_validators=10, n_years=1, n_trials=10))

    # At 5 years, look at KDE for 1/5/50 minipool solitarius/smoothifier
    years = 5
    trials = 1000
    results_ls = []

    input_ls = [
        (1, None, 'solitarius_1'),
        (5, None, 'solitarius_5'),
        (50, None, 'solitarius_50'),
        (1, 2000, '2ksmoothie_1'),
        (5, 2000, '2ksmoothie_5'),
        (50, 2000, '2ksmoothie_50'),
    ]

    for validators, smoothie, label in input_ls:
        res = calc_ppv_ls(
            n_validators=validators,
            n_years=years,
            n_trials=trials,
            n_smoothie=smoothie,
            per_validator=True)
        results_ls.append((res, label))

    # do this first to help select xmax
    # xmin = min([min(resls) for resls in list(zip(*results_ls))[0]])
    # xmax = max([max(resls) for resls in list(zip(*results_ls))[0]])

    xmin, xmax = 0, 3e37  # setting manually
    xpts = np.linspace(xmin, xmax, num=2000)
    fig, subplots = plt.subplots(2, sharex='all')
    for res, lbl in results_ls:
        res = np.clip(res, None, xmax)
        ypts = scipy.stats.gaussian_kde(res).evaluate(xpts)
        subplots.flat[0].plot(xpts, ypts, label=lbl)
        subplots.flat[1].semilogy(xpts, ypts, label=lbl)
    subplots.flat[0].legend()
    plt.show()
