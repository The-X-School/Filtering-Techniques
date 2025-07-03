def simple_reward_function(text_samples, quality_threshold=0.2, length_min=50, length_max=2000, fc_weight=1.0):
    if not text_samples:
        return 0.0
    fc_patterns = ['function', 'call', 'api', 'execute', 'def', '(', ')', '=', 'return']
    
    filtered_samples = []
    total_score = 0.0
    
    for text in text_samples:
        if len(text) < length_min or len(text) > length_max:
            continue
            
        words = text.lower().split()
        if not words:
            continue
            
        length_score = min(1.0, len(text) / 500)
        
        unique_words = set(words)
        vocab_diversity = len(unique_words) / len(words)
        
        fc_matches = sum(1 for pattern in fc_patterns if pattern in text.lower())
        fc_score = min(1.0, fc_matches / len(fc_patterns))
        
        quality_score = (
            length_score * 0.3 +
            vocab_diversity * 0.4 + 
            fc_score * fc_weight * 0.3
        )
        
        if quality_score >= quality_threshold:
            filtered_samples.append(text)
            total_score += quality_score
    
    if len(filtered_samples) < 3:
        return 0.0
    
    avg_score = total_score / len(filtered_samples)
    size_bonus = min(0.2, len(filtered_samples) / 100)
    
    return min(1.0, avg_score + size_bonus)


if __name__ == "__main__":
    sample_texts = [
        "def call_api(endpoint, params): return requests.get(endpoint, params=params)",
        "Execute database query with parameters: db.execute('SELECT * FROM users')",
        "This is just regular text without any special patterns",
        "function() method call with arguments and return values",
        "Short text",
        "API function call example with error handling and response parsing"
    ]
    
    reward = simple_reward_function(sample_texts)
    print(f"Reward score: {reward:.3f}")
    
    reward2 = simple_reward_function(sample_texts, quality_threshold=0.1, fc_weight=2.0)
    print(f"Reward score with different params: {reward2:.3f}") 