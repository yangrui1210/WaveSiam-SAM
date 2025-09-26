import numpy as np

class entropyWeightsScore:
    def __init__(self, non_building, no_damage, minor_damage, major_damage, destroyed):
        self.non_building = non_building
        self.no_damage = no_damage
        self.minor_damage = minor_damage
        self.major_damage = major_damage
        self.destroyed = destroyed


    def calculate_weight_score(self):
        # Step 1: 统计像元数量
        # 这里设置一个样例像元数量，实际使用中请根据真实数据替换
        pixel_counts = {
            0: self.non_building,
            1: self.no_damage,
            2: self.minor_damage,
            3: self.major_damage,
            4: self.destroyed
        }

        # Step 2: 获取损毁等级和像元数量
        categories = list(pixel_counts.keys())
        counts = np.array(list(pixel_counts.values()))

        # Step 3: 归一化处理
        total_counts = np.sum(counts)
        normalized_counts = counts / total_counts

        # Step 4: 计算熵值
        def calculate_entropy(normalized_counts):
            # 避免log(0)出现
            non_zero_counts = normalized_counts[normalized_counts > 0]
            k = 1 / np.log(len(normalized_counts))
            entropy = -k * np.sum(non_zero_counts * np.log(non_zero_counts))
            return entropy

        entropies = np.array([calculate_entropy(normalized_counts)])

        # Step 5: 计算权重
        weights = (1 - entropies) / np.sum(1 - entropies)

        # Step 6: 计算加权得分
        weighted_scores = normalized_counts * weights

        # Step 7: 确定最终损毁等级
        final_damage_level_index = np.argmax(weighted_scores)
        final_damage_level = categories[final_damage_level_index]

        return final_damage_level
