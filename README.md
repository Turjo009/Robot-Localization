Robot Localization

Robot localization is the process of determining where a mobile robot is located concerning its environment. A map of the environment is available, and the robot is equipped with sensors that observe the environment as well as monitor its own motion.

Sensor Model

    The sensors' error rate is ε, and errors occur independently for the four sensors (north/south/west/east).

Goal

    Find the robot's location and the most probable path through a sequence of sensor readings.

Assumptions

    The initial state of the robot is uniformly distributed in all traversable positions.

Output

    A list of maps (matrices) with each element representing the probability of the robot being at that position at that time step.

2D Input Format

4 10                  >> the size of the map (rows by columns)
XXXXXXXXXX          >> map ('X' denotes an obstacle; 'O' represents a traversable position)
XOOOOOOXXX
XOOOOOOXXX
XXXXXXXXXX
4                    >> the number of sensor observations
1011                  >> the observed values (in order NSWE; '1' means obstacle)
1010
1000
1100
0.2                  >> sensor's error rate

Error Probability

If (d_{it}) denotes the number of directions reporting erroneous values, then the probability that a robot at position (i) would receive a sensor reading (e_t) is:

P(E_t = e_t | X_t = i) = (1 - ε)^(4 - d_{it}) * ε^(d_{it})

Viterbi Algorithm

The viterbi.py script can be called as: $ python viterbi.py [input]
