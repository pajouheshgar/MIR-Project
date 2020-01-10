import json

import numpy as np

if __name__ == '__main__':
    data = json.load(open("Data/Phase3/crawler2.json", encoding='utf-8'))[:5000]

    node2index = {x: i for i, x in enumerate((map(lambda x: x['id'], data)))}
    node2title = {paper_id: title for paper_id, title in (map(lambda x: (x['id'], x['title']), data))}
    incoming_neighbors = [[] for _ in node2index]
    outgoing_neighbors = [[] for _ in node2index]
    for row in data:
        paper_id = row['id']
        index = node2index[paper_id]
        for ref in row['references']:
            if ref in node2index:
                incoming_neighbors[node2index[ref]].append(index)
                outgoing_neighbors[index].append(node2index[ref])

    n = len(node2index)
    p = 1 / n
    old_probs = np.array([p for _ in node2index])
    print("Enter Alpha: ")
    alpha = float(input())
    for steps in range(100):
        new_probs = old_probs.copy()
        for node in node2index:
            index = node2index[node]
            new_prob = (1 - alpha)
            for neighbor_index in incoming_neighbors[index]:
                new_prob += alpha * old_probs[neighbor_index] / len(outgoing_neighbors[neighbor_index])

            new_probs[index] = new_prob

        new_probs = new_probs / np.sum(new_probs)
        diff = np.sum(np.abs(old_probs - new_probs))
        old_probs = new_probs
        if diff < 1e-15:
            break

    output = ''
    for paper_id in node2index:

        s = "{}: {} : {}".format(paper_id, node2title[paper_id], old_probs[node2index[paper_id]])
        output += (s + '\n')

        print(s)

    with open("Data/Phase3/page_rank_result.txt", 'w', encoding='utf-8') as f:
        f.write(output)