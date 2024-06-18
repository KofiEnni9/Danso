import numpy as np
import logging
import pickle

# Configure logging
logging.basicConfig(filename='linucb.log', level=logging.INFO, format='%(asctime)s - %(message)s')


def save_context(context, filename="context.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(context, f)

def load_context(filename="context.pkl"):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


# List of subjects
subjects = [
    'contest 1', 'contest 2', 'contest 3', 'contest 4', 'contest 5', 'contest 6', 'contest 7', 'contest 8', 'contest 9', 'contest 10',
    'contest 11', 'contest 12', 'contest 13', 'contest 14', 'contest 15', 'contest 16', 'contest 17', 'contest 18', 'contest 19', 'contest 20',
    'contest 21', 'contest 22', 'contest 23', 'contest 24', 'contest 25', 'contest 26', 'contest 27', 'contest 28', 'contest 29', 'contest 30',
    'contest 31', 'contest 32', 'contest 33', 'contest 34', 'contest 35', 'contest 36', 'contest 37', 'contest 38', 'contest 39', 'contest 40'
]


class LinUCB:
    def __init__(self, n_actions, n_features, alpha=1.0):
        self.n_actions = n_actions
        self.n_features = n_features
        self.alpha = alpha

        # Initialize parameters
        self.A = np.array([np.identity(n_features) for _ in range(n_actions)])  # action covariance matrix
        self.b = np.array([np.zeros(n_features) for _ in range(n_actions)])  # action reward vector
        self.theta = np.array([np.zeros(n_features) for _ in range(n_actions)])  # action parameter vector

        # Initialize interaction counts
        self.interaction_counts = np.zeros(n_actions)

    def predict(self, context):
        context = np.array(context)  # Convert list to ndarray
        p = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            self.theta[a] = np.dot(np.linalg.inv(self.A[a]), self.b[a])  # theta_a = A_a^-1 * b_a
            p[a] = np.dot(self.theta[a], context) + self.alpha * np.sqrt(np.dot(context, np.dot(np.linalg.inv(self.A[a]), context)))
        return p

    def update(self, action, context, reward):
        context = np.array(context)  # Convert list to ndarray if necessary
        context = context.reshape(-1)  # Ensure context is a flat array
        self.A[action] += np.outer(context, context)  # A_a = A_a + x_t * x_t^T
        self.b[action] += reward * context  # b_a = b_a + r_t * x_t
        self.interaction_counts[action] += 1  # Increment interaction count for the chosen action

        # Log the update
        logging.info(f"Action: {action}, Context: {context.tolist()}, Reward: {reward}")
        self.save_state()

    def save_state(self, filename="model_state.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_state(filename="model_state.pkl"):
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None


# Example usage

# Suppose we have 40 subjects (actions) and each context is a 4-dimensional feature vector
n_actions = len(subjects)
n_features = 1
alpha = 1.0

# Try to load an existing model, otherwise initialize a new one
model = LinUCB.load_state() or LinUCB(n_actions, n_features, alpha)

# Example context vector for a user (e.g., user's preferences or history in some feature space)
context = [1, 1, 1, 1]

for each in range(50):
    # Predict the preference scores for each subject
    preference_scores = model.predict(context)
    print("Preference scores:", preference_scores)

    # Select the action (subject) with the highest score
    chosen_action = np.argmax(preference_scores)
    print("Chosen subject based on preference:", subjects[chosen_action])

    # print(type(subjects.index(subjects[chosen_action])))
    # Update the model with the chosen action, context, and reward (e.g., user clicked on the subject)
    answer = "correct"
    if answer == "incorrect":
        reward = 1
    if answer == "correct":
            reward = 0
    model.update(chosen_action, context, reward)
