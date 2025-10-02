import numpy as np
import math
from typing import List, Tuple

def line_to_canonical(line: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
    """
    Преобразует линию из формата двух точек в каноническую форму Ax + By + C = 0
    
    Параметры:
    - line: кортеж (x1, y1, x2, y2)
    
    Возвращает:
    - (A, B, C): коэффициенты канонического уравнения
    """
    x1, y1, x2, y2 = line
    
    # Вычисляем коэффициенты прямой
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    
    # Нормализуем коэффициенты
    norm = math.sqrt(A**2 + B**2)
    if norm > 0:
        A /= norm
        B /= norm
        C /= norm
    
    return A, B, C

def point_to_line_distance(point: Tuple[float, float], line: Tuple[float, float, float]) -> float:
    """
    Вычисляет расстояние от точки до линии в канонической форме
    
    Параметры:
    - point: кортеж (x, y)
    - line: кортеж (A, B, C) канонического уравнения
    
    Возвращает:
    - distance: расстояние от точки до линии
    """
    x, y = point
    A, B, C = line
    return abs(A * x + B * y + C) / math.sqrt(A**2 + B**2)

def line_center(line: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    Вычисляет центр линии
    """
    x1, y1, x2, y2 = line
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def line_angle(line: Tuple[float, float, float]) -> float:
    """
    Вычисляет угол линии в радианах [0, pi)
    """
    A, B, C = line
    if B == 0:
        return math.pi / 2  # Вертикальная линия
    return math.atan2(-A, B) % math.pi

def lines_distance(line1_canonical: Tuple[float, float, float], 
                  line2_canonical: Tuple[float, float, float],
                  line1_points: Tuple[float, float, float, float] = None,
                  line2_points: Tuple[float, float, float, float] = None,
                  method: str = 'combined') -> float:
    """
    Вычисляет расстояние между двумя линиями в канонической форме
    
    Параметры:
    - line1_canonical, line2_canonical: канонические формы линий (A, B, C)
    - line1_points, line2_points: исходные точки линий (для некоторых методов)
    - method: метод вычисления расстояния:
      'parallel' - расстояние между параллельными линиями
      'angle' - угловое расстояние
      'center' - расстояние между центрами линий
      'combined' - комбинированная метрика
      'hausdorff' - расстояние Хаусдорфа (требует точки)
    
    Возвращает:
    - distance: расстояние между линиями
    """
    A1, B1, C1 = line1_canonical
    A2, B2, C2 = line2_canonical
    
    if method == 'parallel':
        # Расстояние между параллельными линиями
        # Проверяем параллельность (скалярное произведение нормалей ≈ 1)
        dot_product = abs(A1 * A2 + B1 * B2)
        if dot_product < 0.95:  # Линии не параллельны
            return float('inf')
        
        # Для параллельных линий расстояние |C2 - C1| / sqrt(A^2 + B^2)
        # Но так как коэффициенты нормализованы, sqrt(A^2 + B^2) = 1
        return abs(C2 - C1)
    
    elif method == 'angle':
        # Угловое расстояние
        angle1 = line_angle(line1_canonical)
        angle2 = line_angle(line2_canonical)
        angle_diff = min(abs(angle1 - angle2), math.pi - abs(angle1 - angle2))
        return angle_diff * 100  # Масштабируем для сравнения с другими метриками
    
    elif method == 'center':
        # Расстояние между центрами линий
        if line1_points is None or line2_points is None:
            raise ValueError("Для метода 'center' нужны исходные точки линий")
        
        center1 = line_center(line1_points)
        center2 = line_center(line2_points)
        dx = center1[0] - center2[0]
        dy = center1[1] - center2[1]
        return math.sqrt(dx**2 + dy**2)
    
    elif method == 'combined':
        # Комбинированная метрика, учитывающая угол и смещение
        if line1_points is None or line2_points is None:
            raise ValueError("Для метода 'combined' нужны исходные точки линий")
        
        # Угловое расстояние
        angle1 = line_angle(line1_canonical)
        angle2 = line_angle(line2_canonical)
        angle_diff = min(abs(angle1 - angle2), math.pi - abs(angle1 - angle2))
        
        # Расстояние между центрами
        center1 = line_center(line1_points)
        center2 = line_center(line2_points)
        center_distance = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        # Расстояние от центров до противоположных линий
        dist1_to_2 = point_to_line_distance(center1, line2_canonical)
        dist2_to_1 = point_to_line_distance(center2, line1_canonical)
        line_distance = (dist1_to_2 + dist2_to_1) / 2
        
        # Комбинированная метрика (можно настроить веса)
        combined = (angle_diff * 50 +  # вес для угла
                   center_distance * 0.5 +  # вес для расстояния между центрами
                   line_distance * 1.0)  # вес для расстояния между линиями
        
        return combined
    
    elif method == 'hausdorff':
        # Расстояние Хаусдорфа (максимальное из минимальных расстояний)
        if line1_points is None or line2_points is None:
            raise ValueError("Для метода 'hausdorff' нужны исходные точки линий")
        
        # Извлекаем точки из линий
        points1 = [(line1_points[0], line1_points[1]), (line1_points[2], line1_points[3])]
        points2 = [(line2_points[0], line2_points[1]), (line2_points[2], line2_points[3])]
        
        # Вычисляем расстояние Хаусдорфа
        max_min_dist = 0
        for p1 in points1:
            min_dist = min(point_to_line_distance(p1, line2_canonical) for p2 in points2)
            max_min_dist = max(max_min_dist, min_dist)
        
        for p2 in points2:
            min_dist = min(point_to_line_distance(p2, line1_canonical) for p1 in points1)
            max_min_dist = max(max_min_dist, min_dist)
        
        return max_min_dist
    
    else:
        raise ValueError(f"Неизвестный метод: {method}")

def calculate_line_distances(lines: List[Tuple[float, float, float, float]], 
                           method: str = 'combined') -> np.ndarray:
    """
    Вычисляет матрицу попарных расстояний между линиями
    
    Параметры:
    - lines: список линий в формате [(x1, y1, x2, y2), ...]
    - method: метод вычисления расстояния
    
    Возвращает:
    - distance_matrix: матрица N x N расстояний между линиями
    """
    n = len(lines)
    distance_matrix = np.zeros((n, n))
    
    # Преобразуем все линии в каноническую форму
    canonical_lines = [line_to_canonical(line) for line in lines]
    
    # Вычисляем попарные расстояния
    for i in range(n):
        for j in range(i + 1, n):
            distance = lines_distance(
                canonical_lines[i], 
                canonical_lines[j],
                lines[i] if method in ['center', 'combined', 'hausdorff'] else None,
                lines[j] if method in ['center', 'combined', 'hausdorff'] else None,
                method
            )
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    
    return distance_matrix

def find_similar_lines(lines: List[Tuple[float, float, float, float]], 
                      threshold: float = 10.0,
                      method: str = 'combined') -> List[List[int]]:
    """
    Находит группы похожих линий на основе матрицы расстояний
    
    Параметры:
    - lines: список линий
    - threshold: порог для определения похожести
    - method: метод вычисления расстояния
    
    Возвращает:
    - groups: список групп индексов похожих линий
    """
    distance_matrix = calculate_line_distances(lines, method)
    n = len(lines)
    visited = set()
    groups = []
    
    for i in range(n):
        if i in visited:
            continue
        
        # Начинаем новую группу с линии i
        group = [i]
        visited.add(i)
        
        # Ищем все линии, близкие к текущей группе
        for j in range(n):
            if j not in visited and distance_matrix[i, j] <= threshold:
                group.append(j)
                visited.add(j)
        
        groups.append(group)
    
    return groups

def remove_similar_lines(lines: List[Tuple[float, float, float, float]], 
                        threshold: float = 10.0,
                        method: str = 'combined',
                        keep_longest: bool = True) -> List[Tuple[float, float, float, float]]:
    """
    Удаляет похожие линии, оставляя по одной из каждой группы
    
    Параметры:
    - lines: список линий
    - threshold: порог для определения похожести
    - method: метод вычисления расстояния
    - keep_longest: если True, оставляет самую длинную линию в группе
    
    Возвращает:
    - unique_lines: список уникальных линий
    """
    def line_length(line):
        x1, y1, x2, y2 = line
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    groups = find_similar_lines(lines, threshold, method)
    unique_lines = []
    
    for group in groups:
        if keep_longest:
            # Оставляем самую длинную линию в группе
            longest_line = max([lines[i] for i in group], key=line_length)
            unique_lines.append(longest_line)
        else:
            # Оставляем первую линию в группе
            unique_lines.append(lines[group[0]])
    
    return unique_lines