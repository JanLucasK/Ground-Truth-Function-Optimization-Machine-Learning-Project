import matplotlib.pyplot as plt

class ResultVisualizer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_names = []
        self.neural_net_optimum = []
        self.ground_truth_optimum = []
        self.distances = []

    def read_data_from_file(self):
        with open(self.file_path, "r") as file:
            lines = file.readlines()

        current_data = {
            "neural_net_optimum": None,
            "ground_truth_optimum": None,
            "distance": None
        }

        for line in lines:
            if line.startswith("f_"):
                if current_data["neural_net_optimum"] is not None:
                    self.process_data(current_data)
                current_data["file_name"] = line.strip()
            elif line.startswith("neural-net optium:"):
                current_data["neural_net_optimum"] = [float(val) for val in line.split("[")[1].split("]")[0].split()]
            elif line.startswith("ground-thruth optimum:"):
                current_data["ground_truth_optimum"] = [float(val) for val in line.split("[")[1].split("]")[0].split()]
            elif line.startswith("distance:"):
                current_data["distance"] = float(line.split(":")[1].strip())

        if current_data["neural_net_optimum"] is not None:
            self.process_data(current_data)

    def process_data(self, data):
        self.file_names.append(data["file_name"])
        self.neural_net_optimum.append(data["neural_net_optimum"])
        self.ground_truth_optimum.append(data["ground_truth_optimum"])
        self.distances.append(data["distance"])

    def plot_data(self):
        plt.figure(figsize=(12, 8))  # Erstelle ein größeres Koordinatensystem

        for i in range(len(self.file_names)):
            plt.scatter(self.neural_net_optimum[i][0], self.neural_net_optimum[i][1], label=f"Neural Net ({self.file_names[i]})", marker="o")
            if self.ground_truth_optimum[i] is not None:
                plt.scatter(self.ground_truth_optimum[i][0], self.ground_truth_optimum[i][1], label=f"Ground Truth ({self.file_names[i]})", marker="x")

        plt.title("Optimum Visualisierung")
        plt.xlabel("X-Wert")
        plt.ylabel("Y-Wert")
        plt.legend()
        plt.grid(True)
        plt.show()

def main():
    file_path = "pso_results_50SwarmSize.txt"
    data_visualizer = ResultVisualizer(file_path)
    data_visualizer.read_data_from_file()
    data_visualizer.plot_data()

if __name__ == "__main__":
    main()
