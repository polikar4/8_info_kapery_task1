import pandas as pd
import numpy as np
import random

# Чтение CSV файла для матрицы расстояний
def read_distance_matrix_from_csv(file_path):
    df = pd.read_csv(file_path, index_col=0)
    num_nodes = df.shape[0]  # Предполагаем, что матрица квадратная
    distance_matrix = np.zeros((num_nodes, num_nodes))
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            value = df.iloc[i, j]
            if value == "-":
                distance_matrix[i, j] = float('inf')  # Недостижимый маршрут
            else:
                distance_matrix[i, j] = float(value)  # Стоимость перемещения или 0
    
    return distance_matrix

# Класс для моделирования автобуса
class Bus:
    def __init__(self, capacity=10, max_time=15):
        self.capacity = capacity  # Максимальная вместимость
        self.max_time = max_time  # Максимальное время в тактах
        self.route = []           # Маршрут автобуса (список узлов)
        self.current_time = 0     # Текущее время маршрута
        self.groups = []          # Список групп в автобусе
        self.current_capacity = 0 # Текущая занятость
        self.available_tickets = []  # Список доступных билетов для каждой достопримечательности

    def add_group(self, group_size, tickets_for_attraction):
        """Добавить группу в автобус, если есть достаточно билетов для достопримечательности."""
        if self.current_capacity + group_size <= self.capacity:
            # Проверка на доступность билетов для каждого посещаемого узла
            if tickets_for_attraction >= group_size:
                self.groups.append(group_size)
                self.current_capacity += group_size
                return True
        return False

    def add_to_route(self, node, travel_cost):
        self.route.append(node)
        self.current_time += travel_cost

# Инициализация автобусов
buses = [Bus() for _ in range(15)]

# Муравьиный алгоритм
class AntColonyOptimizer:
    def __init__(self, distance_matrix, tickets, num_ants=50, num_iterations=500, alpha=1.0, beta=2.0, evaporation_rate=0.5):
        self.distance_matrix = distance_matrix
        self.tickets = tickets  # Количество билетов для каждой достопримечательности
        self.num_nodes = len(distance_matrix)
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.pheromone = np.ones((self.num_nodes, self.num_nodes))  # Инициализация феромонов
        self.best_route = None
        self.best_cost = float('inf')

    def _calculate_probabilities(self, current_node, visited_nodes, available_nodes):
        pheromone = np.copy(self.pheromone[current_node])
        heuristic = 1.0 / (self.distance_matrix[current_node] + 1e-10)  # Эвристика на основе расстояния
        
        # Создание логической маски для посещённых узлов
        visited_mask = np.zeros(self.num_nodes, dtype=bool)
        visited_mask[list(visited_nodes)] = True

        # Применение маски к феромонам и эвристике
        pheromone[visited_mask] = 0
        heuristic[visited_mask] = 0
        
        # Убираем недоступные узлы (если билетов нет или уже посещены)
        for node in range(self.num_nodes):
            if node not in available_nodes:
                pheromone[node] = 0
                heuristic[node] = 0

        probabilities = (pheromone ** self.alpha) * (heuristic ** self.beta)
        total = np.sum(probabilities)
        if total == 0:
            return np.zeros_like(probabilities)
        return probabilities / total

    def _evaporate_pheromones(self):
        self.pheromone *= (1 - self.evaporation_rate)

    def _update_pheromones(self, route, cost):
        for i in range(len(route) - 1):
            from_node = route[i]
            to_node = route[i + 1]
            self.pheromone[from_node, to_node] += 1.0 / cost

    def find_best_route(self):
        for iteration in range(self.num_iterations):
            routes = []
            costs = []

            for _ in range(self.num_ants):
                route, cost = self._construct_solution()
                routes.append(route)
                costs.append(cost)

                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_route = route

            self._evaporate_pheromones()
            
            for route, cost in zip(routes, costs):
                self._update_pheromones(route, cost)

        return self.best_route, self.best_cost

    def _construct_solution(self):
        current_node = 0  # Начинаем с вокзала
        route = [current_node]
        visited_nodes = set(route)
        total_cost = 0
        available_nodes = set(range(self.num_nodes))  # Все узлы, которые можно посетить

        while len(visited_nodes) < self.num_nodes:
            probabilities = self._calculate_probabilities(current_node, visited_nodes, available_nodes)
            if np.sum(probabilities) == 0:
                break  # Если нет доступных узлов для посещения

            next_node = np.random.choice(range(self.num_nodes), p=probabilities)
            travel_cost = self.distance_matrix[current_node, next_node]

            # Проверка на допустимость посещения достопримечательности
            if travel_cost < float('inf'):
                route.append(next_node)
                visited_nodes.add(next_node)
                total_cost += travel_cost
                available_nodes.remove(next_node)  # Убираем узел из доступных

                current_node = next_node

        # Возвращение на вокзал
        travel_cost = self.distance_matrix[current_node, 0]
        total_cost += travel_cost
        route.append(0)

        return route, total_cost

# Пример использования
file_path = 'path_to_your_file.csv'  # Замените на ваш путь к файлу
distance_matrix = read_distance_matrix_from_csv('node.csv')

# Здесь должны быть билеты для каждого узла
tickets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 5, 5, 2, 9, 3, 5, 2, 3, 3, 2, 9, 5, 9, 3, 3, 2, 3, 2, 5, 2, 5, 3, 5, 3, 5, 2, 2, 2, 2, 5, 9, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Инициализация муравьиного оптимизатора
optimizer = AntColonyOptimizer(distance_matrix, tickets)
# После вызова find_best_route()
best_route, best_cost = optimizer.find_best_route()

# Преобразуем маршрут в стандартные целые числа Python
cleaned_route = [int(node) for node in best_route]

# Выводим результаты
print(f"Лучший маршрут: {cleaned_route}")
print(f"Общая стоимость перемещений: {best_cost}")