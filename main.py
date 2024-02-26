# Naive Bayesian Classification



# Sets up the data input into lists and runs program


def main():
    bird_probabilities = read_probabilities("/Users/bryan.gonzalez/PycharmProjects/Mobi_RT/exercise-mobi-clean-code/pdf.txt")
    plane_probabilities = read_probabilities("/Users/bryan.gonzalez/PycharmProjects/Mobi_RT/exercise-mobi-clean-code/pdf.txt")
    data_array = read_data("/Users/bryan.gonzalez/PycharmProjects/Mobi_RT/exercise-mobi-clean-code/data.txt")

    result = run_naive_bayesian(bird_probabilities, plane_probabilities, data_array)
    pretty_print(result)



# Helper function to print out the result
def pretty_print (result):
    for i in range(0,10):
        print("[", i + 1, "]: ", result[i])

# The main implementation of the naive Bayesian algorithm

def read_probabilities(file_path):
    probabilities = {}
    with open(file_path, 'r') as file:
        line = file.readline().strip('\n').split(',')
        j = 0
        for i in range(0, 400):
            probabilities[j] = float(line[i])
            j += 0.5
    return probabilities


def read_data(file_path):
    data_array = []
    with open(file_path, 'r') as file:
        for _ in range(10):
            data_array.append([])
        for line in file:
            line = line.strip('\n').split(',')
            for k in range(0, 300):
                if line[k] != 'NaN':
                    new = round(float(line[k]) * 2) / 2
                    data_array[len(data_array) - 1].append(new)
    return data_array


def run_naive_bayesian(bird_probabilities, plane_probabilities, data_array):
    classification = []

    for observation in data_array:
        b_plane = [plane_probabilities[observation[0]] * 0.9]
        b_bird = [bird_probabilities[observation[0]] * 0.9]

        for i in range(1, len(observation)):
            plane = b_plane[-1] + plane_probabilities[observation[i]] * (0.9 * b_plane[-1] + 0.1 * b_bird[-1])
            bird = b_bird[-1] + bird_probabilities[observation[i]] * (0.9 * b_bird[-1] + 0.1 * b_plane[-1])
            total_sum = plane + bird
            b_plane.append(plane / total_sum if total_sum != 0 else 0)
            b_bird.append(bird / total_sum if total_sum != 0 else 0)

        plane_num, bird_num = b_plane[-1], b_bird[-1]

        if plane_num > (bird_num + 0.10):
            classification.append(f"Aircraft = {plane_num}")
        elif bird_num > (plane_num + 0.05):
            classification.append(f"Bird = {bird_num}")
        else:
            classification.append("Could not be determined.")

    return classification



if __name__ == "__main__":
    main()