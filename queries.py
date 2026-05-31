"""Hand-written queries and ground truth answers for RAG evaluation."""

QUERIES = [
    {
        "query": "According to Paul Graham, what is the key to doing work you truly love?",
        "type": "factual",
        "ground_truth": "Paul Graham argues that the key to doing work you love is to keep searching and not settle. He explains that most people don't know what they love initially — finding it requires trying many different things, paying attention to what genuinely interests you, and being willing to switch paths when your current work doesn't feel right. He emphasizes that you should follow your curiosity and not let prestige or money be the primary driver of your choices.",
        "relevant_doc_ids": ["how_to_do_what_you_love_chunk_0", "how_to_do_what_you_love_chunk_1"],
    },
    {
        "query": "How does Paul Graham's concept of 'heresy' in 'What You Can't Say' relate to his advice about pursuing unconventional career paths?",
        "type": "multi_hop",
        "ground_truth": "In 'What You Can't Say', Paul Graham discusses how every society has ideas that cannot be expressed — 'heresies' — and that independent thinkers often recognize these forbidden ideas. This connects to his career advice in 'How to Do What You Love', where he encourages pursuing unconventional paths that others might dismiss. The common thread is that valuable insights and fulfilling work often lie outside mainstream acceptance, and the ability to think independently about what is true — even when it contradicts conventional wisdom — is essential for both intellectual honesty and finding meaningful work.",
        "relevant_doc_ids": ["what_you_cant_say_chunk_2", "how_to_do_what_you_love_chunk_3"],
    },
    {
        "query": "Which of the following is NOT a reason Paul Graham gives for why nerds are unpopular in school: (a) they don't care about popularity, (b) they are naturally less intelligent than their peers, (c) the school environment artificially creates a popularity hierarchy, or (d) they focus on interests that don't translate to social status?",
        "type": "negation",
        "ground_truth": "Paul Graham does NOT claim that nerds are naturally less intelligent than their peers — in fact, he argues the opposite. He states that nerds are unpopular because the school environment creates an artificial social hierarchy based on superficial traits, and nerds tend to focus on substantive interests (like programming, reading, or academics) that don't translate to social status in that environment. He also notes that many nerds don't care about popularity because they find fulfillment in their intellectual pursuits.",
        "relevant_doc_ids": ["why_nerds_are_unpopular_chunk_0", "why_nerds_are_unpopular_chunk_1"],
    },
    {
        "query": "Compare Paul Graham's perspective on wealth creation in 'How to Make Wealth' with his views on meaningful work in 'How to Do What You Love'. How do these two essays complement or contradict each other?",
        "type": "comparison",
        "ground_truth": "'How to Make Wealth' argues that wealth is created by building things people want — it's a positive-sum game where innovation benefits everyone. 'How to Do What You Love' argues that the best way to find fulfilling work is to follow genuine curiosity rather than pursuing money. These essays complement rather than contradict each other: Graham suggests that doing what you love often leads to creating value (wealth) naturally. He distinguishes between merely chasing money and creating genuine value — the latter is both more fulfilling and more likely to produce wealth. The contradiction is only apparent if one confuses wealth creation with greed.",
        "relevant_doc_ids": ["how_to_make_wealth_chunk_0", "how_to_make_wealth_chunk_1", "how_to_do_what_you_love_chunk_0"],
    },
    {
        "query": "Summarize Paul Graham's overall philosophy on how people should approach their careers, drawing from his essays on work, wealth, and independent thinking.",
        "type": "summarisation",
        "ground_truth": "Paul Graham's overarching career philosophy centers on authentic engagement: people should pursue work that genuinely interests them rather than optimizing for prestige, money, or social approval. He argues that genuine curiosity leads to better work, which in turn creates real value and often leads to financial success as a byproduct. His essays encourage thinking independently about what matters, questioning societal assumptions about success, and having the courage to follow unconventional paths. The core message is that fulfillment comes from doing work you find intrinsically meaningful, and that this approach is both personally satisfying and economically powerful.",
        "relevant_doc_ids": ["how_to_do_what_you_love_chunk_0", "how_to_do_what_you_love_chunk_2", "what_you_cant_say_chunk_0", "how_to_make_wealth_chunk_0"],
    },
]
