# hypergraph_processor.py
import numpy as np
import torch
import dgl
import hypernetx as hnx
import matplotlib.pyplot as plt
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from torch import nn
import torch.nn.functional as F

class HypergraphProcessor:
    def __init__(self):
        self.lda_model = None
        self.dictionary = None
        self.hypergraph = None

    def create_lda_model(self, processed_text, num_topics=8, passes=15):
        """Train and validate LDA model"""
        self.dictionary = corpora.Dictionary(processed_text)
        corpus = [self.dictionary.doc2bow(text) for text in processed_text]
        
        self.lda_model = LdaModel(
            corpus=corpus,
            id2word=self.dictionary,
            num_topics=num_topics,
            passes=passes,
            random_state=42
        )
        return self.lda_model

    def optimize_lda(self, processed_text, max_topics=40, step=5):
        """End-to-end LDA optimization pipeline"""
        self.dictionary = corpora.Dictionary(processed_text)
        corpus = [self.dictionary.doc2bow(text) for text in processed_text]
        
        model_list, coherence_values = self._compute_coherence_values(
            corpus, processed_text, max_topics, step
        )
        
        # Visualization
        x = range(2, max_topics, step)
        plt.plot(x, coherence_values)
        plt.xlabel("Number of Topics")
        plt.ylabel("Coherence Score")
        plt.title("Topic Coherence Optimization")
        plt.show()
        
        # Get optimal parameters
        best_idx = np.argmax(coherence_values)
        optimal_topics = x[best_idx]
        best_model, best_alpha, best_beta = self._tune_hyperparameters(
            corpus, processed_text, optimal_topics
        )
        
        self.lda_model = best_model
        return {
            'topics': optimal_topics,
            'alpha': best_alpha,
            'beta': best_beta,
            'coherence': coherence_values[best_idx]
        }

    def build_hypergraph(self):
        """Construct hypernetx graph from LDA topics"""
        if not self.lda_model:
            raise ValueError("Train LDA model first using create_lda_model()")
            
        hyperedges = {
            f"Topic_{i}": {word for word, _ in self.lda_model.show_topic(i)}
            for i in range(self.lda_model.num_topics)
        }
        self.hypergraph = hnx.Hypergraph(hyperedges)
        return self.hypergraph

    def visualize_hypergraph(self, title="Research Paper Topic Hypergraph"):
        """Draw interactive hypergraph visualization"""
        if not self.hypergraph:
            raise ValueError("Build hypergraph first using build_hypergraph()")
            
        hnx.drawing.draw(self.hypergraph)
        plt.title(title)
        plt.show()

    def _compute_coherence_values(self, corpus, texts, max_topics, step):
        """Calculate coherence scores for different topic counts"""
        coherence_values = []
        model_list = []
        
        for num_topics in range(2, max_topics, step):
            model = LdaModel(
                corpus=corpus,
                id2word=self.dictionary,
                num_topics=num_topics,
                random_state=42,
                passes=15,
                alpha='auto',
                eta='auto'
            )
            model_list.append(model)
            
            coherence = CoherenceModel(
                model=model,
                texts=texts,
                dictionary=self.dictionary,
                coherence='c_v'
            ).get_coherence()
            
            coherence_values.append(coherence)
            print(f"Topics: {num_topics} | Coherence: {coherence:.3f}")
            
        return model_list, coherence_values

    def _tune_hyperparameters(self, corpus, texts, num_topics):
        """Optimize alpha and beta parameters"""
        best_coherence = -1
        best_params = {}
        best_model = None
        
        alpha_values = list(np.round(np.arange(0.01, 1, 0.3), 2)) + ['symmetric', 'asymmetric']
        beta_values = list(np.round(np.arange(0.01, 1, 0.3), 2)) + ['auto']
        
        for alpha in alpha_values:
            for beta in beta_values:
                try:
                    model = LdaModel(
                        corpus=corpus,
                        id2word=self.dictionary,
                        num_topics=num_topics,
                        alpha=alpha,
                        eta=beta,
                        passes=25,
                        random_state=42
                    )
                    
                    coherence = CoherenceModel(
                        model=model,
                        texts=texts,
                        dictionary=self.dictionary,
                        coherence='c_v'
                    ).get_coherence()
                    
                    if coherence > best_coherence:
                        best_coherence = coherence
                        best_params = {'alpha': alpha, 'beta': beta}
                        best_model = model
                        
                        print(f"New best: α={alpha}, β={beta} | Coherence: {coherence:.3f}")
                        
                except Exception as e:
                    continue
                    
        return best_model, best_params['alpha'], best_params['beta']

class HypergraphConv(nn.Module):
    """Hypergraph convolution layer"""
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.weight = nn.Linear(in_feats, out_feats)
        
    def forward(self, g, features):
        with g.local_scope():
            # Explicitly set features for node type
            g.nodes['node'].data['h'] = features
            
            # Step 1: Node -> Edge aggregation
            g.update_all(
                dgl.function.copy_u('h', 'm'),
                dgl.function.mean('m', 'h_edge'),
                etype='in'
            )
            
            # Step 2: Edge -> Node aggregation
            g.update_all(
                dgl.function.copy_u('h_edge', 'm'),
                dgl.function.mean('m', 'h_node'),
                etype='contains'
            )
            
            return self.weight(g.nodes['node'].data['h_node'])

class HyperGNN:
    def __init__(self, hypergraph, lda_model):
        self.hypergraph = hypergraph
        self.lda_model = lda_model
        self.node_list = sorted(hypergraph.nodes)
        
    def train(self, num_epochs=100):
        """Train HyperGNN on the hypergraph structure"""
        g = self._convert_to_dgl()
        node_features = self._create_node_features()
        
        model = HypergraphConv(self.lda_model.num_topics, 32)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            embeddings = model(g, node_features)
            loss = self._contrastive_loss(embeddings)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss {loss.item():.4f}")
            
        return model, embeddings

    def _convert_to_dgl(self):
        """Convert hypernetx hypergraph to DGL heterograph"""
        node_list = self.node_list
        edge_list = list(self.hypergraph.edges)
        
        src, dst = [], []
        for edge_id, members in self.hypergraph.incidence_dict.items():
            for node in members:
                src.append(node_list.index(node))
                dst.append(edge_list.index(edge_id))
        
        return dgl.heterograph({
            ('node', 'in', 'edge'): (src, dst),
            ('edge', 'contains', 'node'): (dst, src)
        })

    def _create_node_features(self):
        """Initialize node features using LDA topic distributions"""
        node_features = torch.zeros(len(self.node_list), self.lda_model.num_topics)
        for idx, node in enumerate(self.node_list):
            topics = self.lda_model.get_term_topics(node) or []
            for topic, prob in topics:
                node_features[idx, topic] = float(prob)
        return node_features

    def _contrastive_loss(self, embeddings, margin=1.0):
        """Calculate contrastive loss"""
        node_list = self.node_list
        num_nodes = len(node_list)
        
        # Positive pairs (same hyperedge)
        pos_pairs = []
        for edge in self.hypergraph.edges:
            members = list(self.hypergraph.edges[edge])
            indices = [node_list.index(n) for n in members]
            pos_pairs.extend([(i,j) for i in indices for j in indices if i != j])
        
        # Negative pairs (different hyperedges)
        neg_pairs = []
        for i in range(num_nodes):
            neighbors = set()
            for edge in self.hypergraph.edges:
                if node_list[i] in self.hypergraph.edges[edge]:
                    neighbors.update(self.hypergraph.edges[edge])
            non_neighbors = [node_list.index(n) for n in node_list 
                            if n not in neighbors and n != node_list[i]]
            neg_pairs.extend([(i,j) for j in non_neighbors])
        
        if not pos_pairs or not neg_pairs:
            return torch.tensor(0.0, requires_grad=True)
        
        # Calculate losses
        pos_loss = F.pairwise_distance(
            embeddings[torch.tensor([i for i,j in pos_pairs])],
            embeddings[torch.tensor([j for i,j in pos_pairs])]
        ).mean()
        
        neg_loss = F.relu(margin - F.pairwise_distance(
            embeddings[torch.tensor([i for i,j in neg_pairs])],
            embeddings[torch.tensor([j for i,j in neg_pairs])]
        )).mean()
        
        return pos_loss + neg_loss