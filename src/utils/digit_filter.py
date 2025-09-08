from torch.utils.data import Subset
from typing import List

class DigitFilter:
    """Utility class to filter datasets by digit groups."""
    
    @staticmethod
    def filter_by_digits(dataset, target_digits: List[int]) -> Subset:
        """
        Filter dataset to include only specified digits.
        
        Args:
            dataset: PyTorch dataset
            target_digits: List of digits to include (e.g., [0,2,4,6,8])
        
        Returns:
            Subset containing only samples with target digits
        """
        indices = []
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            if label in target_digits:
                indices.append(idx)
        
        return Subset(dataset, indices)
    
    @staticmethod
    def get_even_digits() -> List[int]:
        """Get list of even digits."""
        return [0, 2, 4, 6, 8]
    
    @staticmethod
    def get_odd_digits() -> List[int]:
        """Get list of odd digits."""
        return [1, 3, 5, 7, 9]
    
    @staticmethod
    def get_all_digits() -> List[int]:
        """Get list of all digits."""
        return list(range(10))

