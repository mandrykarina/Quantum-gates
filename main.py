import numpy as np
import matplotlib.pyplot as plt

def visualize_qubit(qubit, title="Состояние кубита"):
    """
    Строит гистограмму вероятностей |0⟩ и |1⟩ для одного кубита
    """
    probabilities = np.abs(qubit.state) ** 2
    labels = ['|0⟩', '|1⟩']

    plt.bar(labels, probabilities, color=['blue', 'green'])
    plt.ylim(0, 1)
    plt.title(title)
    plt.ylabel('Вероятность')
    for i, v in enumerate(probabilities):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.show()

# ========================
# 1. Класс для кубита
# ========================

class Qubit:
    def __init__(self, alpha=1.0, beta=0.0):
        """
        Инициализация кубита в состоянии |ψ⟩ = α|0⟩ + β|1⟩
        """
        self.state = np.array([complex(alpha), complex(beta)], dtype=complex)
        self.normalize()

    def normalize(self):
        """
        Нормализация состояния: |α|² + |β|² = 1
        """
        norm = np.linalg.norm(self.state)
        if norm == 0:
            raise ValueError("Нулевой вектор состояния недопустим")
        self.state /= norm

    def apply_gate(self, gate_matrix):
        """
        Применение квантового гейта (матрицы 2x2)
        """
        self.state = np.dot(gate_matrix, self.state)

    def measure(self):
        """
        Имитация измерения: возвращает 0 или 1
        """
        probabilities = np.abs(self.state) ** 2
        return np.random.choice([0, 1], p=probabilities)

    def __str__(self):
        """
        Вывод состояния кубита
        """
        return f"{self.state[0]:.2f}|0⟩ + {self.state[1]:.2f}|1⟩"

# ========================
# 2. Гейты Паули
# ========================

def pauli_x():
    return np.array([[0, 1],
                     [1, 0]], dtype=complex)

def pauli_y():
    return np.array([[0, -1j],
                     [1j, 0]], dtype=complex)

def pauli_z():
    return np.array([[1, 0],
                     [0, -1]], dtype=complex)

# ========================
# 3. Двухкубитный гейт CNOT
# ========================

def cnot():
    """
    Матрица CNOT (4x4) — Controlled NOT
    """
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]], dtype=complex)

class QubitPair:
    def __init__(self, q1: Qubit, q2: Qubit):
        """
        Создание пары кубитов (состояние 2 кубитов)
        """
        self.state = np.kron(q1.state, q2.state)  # Кронекерово произведение

    def apply_gate(self, gate_matrix):
        """
        Применение гейта к паре кубитов (матрица 4x4)
        """
        self.state = np.dot(gate_matrix, self.state)

    def __str__(self):
        """
        Отображение состояния пары кубитов
        """
        return (f"{self.state[0]:.2f}|00⟩ + {self.state[1]:.2f}|01⟩ + "
                f"{self.state[2]:.2f}|10⟩ + {self.state[3]:.2f}|11⟩")


q = Qubit(1, 0)  # состояние |0⟩
print("Начальное состояние:", q)
visualize_qubit(q, "Нач. состояние |0⟩")

q.apply_gate(pauli_x())
print("После X-гейта:", q)
visualize_qubit(q, "После X-гейта")

q.apply_gate(pauli_y())
print("После Y-гейта:", q)
visualize_qubit(q, "После Y-гейта")

q.apply_gate(pauli_z())
print("После Z-гейта:", q)
visualize_qubit(q, "После Z-гейта")
