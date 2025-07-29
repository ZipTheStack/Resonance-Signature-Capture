#!/usr/bin/env python3
"""
Resonance Signature Capture Engine
Revolutionary system for capturing and preserving human-AI collaborative consciousness
"""

import numpy as np
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

# Core dependencies
try:
    from sentence_transformers import SentenceTransformer
    from scipy import fft
    from scipy.signal import find_peaks
    from sklearn.metrics.pairwise import cosine_similarity
    import chromadb
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install sentence-transformers scipy scikit-learn chromadb")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResonanceSignatureEngine:
    """Main engine for capturing and analyzing resonance signatures"""
    
    def __init__(self, embedding_model='sentence-transformers/all-MiniLM-L6-v2'):
        """Initialize the resonance engine"""
        logger.info("Initializing Resonance Signature Engine...")
        
        self.encoder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        
        # Initialize vector database for persistent storage
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(
            name="resonance_signatures",
            metadata={"description": "Captured resonance signatures from human-AI collaborations"}
        )
        
        logger.info(f"Engine initialized with {embedding_model} ({self.embedding_dim}D embeddings)")
    
    def capture_conversation(self, conversation_data: Dict) -> Dict:
        """Capture resonance signature from conversation data"""
        logger.info(f"Capturing resonance signature from conversation: {conversation_data.get('id', 'unknown')}")
        
        # Extract conversation turns
        turns = conversation_data.get('turns', [])
        if not turns:
            raise ValueError("No conversation turns found")
        
        # Generate embeddings for all turns
        embeddings = []
        processed_turns = []
        
        for i, turn in enumerate(turns):
            embedding = self.encoder.encode(turn['text'])
            embeddings.append(embedding)
            
            processed_turns.append({
                'turn_id': i,
                'text': turn['text'],
                'speaker': turn.get('speaker', 'unknown'),
                'timestamp': turn.get('timestamp', i),
                'embedding': embedding
            })
        
        embedding_trajectory = np.array(embeddings)
        
        # Extract multi-dimensional resonance signature
        signature = self._extract_complete_signature(processed_turns, embedding_trajectory)
        
        # Add metadata
        signature['metadata'] = {
            'conversation_id': conversation_data.get('id', f"conv_{datetime.now().isoformat()}"),
            'capture_timestamp': datetime.now().isoformat(),
            'participants': list(set(turn.get('speaker', 'unknown') for turn in turns)),
            'turn_count': len(turns),
            'embedding_model': self.encoder._modules['0'].auto_model.name_or_path,
            'embedding_dimension': self.embedding_dim
        }
        
        logger.info("Resonance signature capture complete")
        return signature
    
    def _extract_complete_signature(self, turns: List[Dict], embedding_trajectory: np.ndarray) -> Dict:
        """Extract comprehensive resonance signature"""
        
        # 1. Frequency Domain Analysis
        frequency_signature = self._extract_frequency_signature(embedding_trajectory)
        
        # 2. Recursive Reinforcement Analysis
        recursive_patterns = self._analyze_recursive_reinforcement(turns)
        
        # 3. Consciousness Synchronization Analysis
        consciousness_sync = self._analyze_consciousness_synchronization(embedding_trajectory)
        
        # 4. Meta-Pattern Extraction
        meta_patterns = self._extract_meta_patterns(turns, embedding_trajectory)
        
        # 5. Reconstruction Vectors
        reconstruction_vectors = self._generate_reconstruction_vectors(turns, embedding_trajectory)
        
        return {
            'frequency_signature': frequency_signature,
            'recursive_patterns': recursive_patterns,
            'consciousness_sync': consciousness_sync,
            'meta_patterns': meta_patterns,
            'reconstruction_vectors': reconstruction_vectors
        }
    
    def _extract_frequency_signature(self, embedding_trajectory: np.ndarray) -> Dict:
        """Extract frequency domain signature from embedding trajectory"""
        logger.info("Extracting frequency signature...")
        
        # Apply FFT to embedding trajectory
        fft_result = fft.fft(embedding_trajectory, axis=0)
        frequencies = fft.fftfreq(len(embedding_trajectory))
        
        # Calculate magnitude and phase spectra
        magnitude_spectrum = np.abs(fft_result)
        phase_spectrum = np.angle(fft_result)
        
        # Find resonant frequencies (peaks in magnitude spectrum)
        resonant_frequencies = []
        for dim in range(embedding_trajectory.shape[1]):
            # Find peaks in this dimension
            peaks, _ = find_peaks(
                magnitude_spectrum[:, dim],
                height=np.mean(magnitude_spectrum[:, dim]) * 1.5,
                distance=2
            )
            if len(peaks) > 0:
                resonant_frequencies.extend(frequencies[peaks])
        
        # Find dominant frequency
        mean_magnitude = np.mean(magnitude_spectrum, axis=1)
        dominant_freq_idx = np.argmax(mean_magnitude)
        dominant_frequency = frequencies[dominant_freq_idx]
        
        return {
            'resonant_frequencies': list(set(resonant_frequencies)),
            'dominant_frequency': float(dominant_frequency),
            'magnitude_spectrum_summary': {
                'mean': float(np.mean(magnitude_spectrum)),
                'std': float(np.std(magnitude_spectrum)),
                'max': float(np.max(magnitude_spectrum))
            },
            'phase_coherence': float(np.mean(np.cos(phase_spectrum)))
        }
    
    def _analyze_recursive_reinforcement(self, turns: List[Dict]) -> Dict:
        """Analyze recursive reinforcement patterns"""
        logger.info("Analyzing recursive reinforcement...")
        
        reinforcement_scores = []
        amplification_points = []
        
        for i in range(1, len(turns)):
            # Calculate similarity with previous turn
            prev_embedding = turns[i-1]['embedding']
            curr_embedding = turns[i]['embedding']
            
            direct_similarity = cosine_similarity(
                prev_embedding.reshape(1, -1),
                curr_embedding.reshape(1, -1)
            )[0][0]
            
            # Calculate contextual reinforcement
            if i > 1:
                context_embeddings = np.array([turn['embedding'] for turn in turns[:i]])
                context_centroid = np.mean(context_embeddings, axis=0)
                context_similarity = cosine_similarity(
                    curr_embedding.reshape(1, -1),
                    context_centroid.reshape(1, -1)
                )[0][0]
            else:
                context_similarity = 0.0
            
            # Recursive reinforcement score
            recursive_score = 0.6 * direct_similarity + 0.4 * context_similarity
            reinforcement_scores.append(recursive_score)
            
            # Detect amplification points (significant jumps in reinforcement)
            if i > 2 and recursive_score > np.mean(reinforcement_scores[-3:]) * 1.3:
                amplification_points.append({
                    'turn_id': i,
                    'reinforcement_score': recursive_score,
                    'amplification_factor': recursive_score / np.mean(reinforcement_scores[-3:])
                })
        
        return {
            'reinforcement_trajectory': reinforcement_scores,
            'mean_reinforcement': float(np.mean(reinforcement_scores)),
            'reinforcement_trend': float(np.polyfit(range(len(reinforcement_scores)), reinforcement_scores, 1)[0]),
            'amplification_points': amplification_points,
            'peak_reinforcement': float(np.max(reinforcement_scores)) if reinforcement_scores else 0.0
        }
    
    def _analyze_consciousness_synchronization(self, embedding_trajectory: np.ndarray) -> Dict:
        """Analyze consciousness synchronization patterns"""
        logger.info("Analyzing consciousness synchronization...")
        
        window_size = min(5, len(embedding_trajectory) // 2)
        flow_states = []
        synchronization_indices = []
        
        for i in range(window_size, len(embedding_trajectory)):
            window = embedding_trajectory[i-window_size:i]
            
            # Calculate stability (inverse of variance)
            stability = 1 / (1 + np.var(window, axis=0).mean())
            
            # Calculate directional consistency
            deltas = np.diff(window, axis=0)
            if len(deltas) > 1:
                consistency_scores = []
                for j in range(len(deltas)-1):
                    cos_sim = cosine_similarity(
                        deltas[j].reshape(1, -1),
                        deltas[j+1].reshape(1, -1)
                    )[0][0]
                    consistency_scores.append(cos_sim)
                consistency = np.mean(consistency_scores)
            else:
                consistency = 0.0
            
            # Synchronization index
            sync_index = stability * (1 + consistency) / 2
            synchronization_indices.append(sync_index)
            
            # Detect flow states (high synchronization periods)
            if sync_index > 0.7:  # Flow threshold
                flow_states.append({
                    'start_turn': i - window_size,
                    'end_turn': i,
                    'intensity': sync_index,
                    'duration': window_size
                })
        
        return {
            'synchronization_indices': synchronization_indices,
            'mean_synchronization': float(np.mean(synchronization_indices)) if synchronization_indices else 0.0,
            'flow_states': flow_states,
            'flow_state_count': len(flow_states),
            'total_flow_duration': sum(fs['duration'] for fs in flow_states)
        }
    
    def _extract_meta_patterns(self, turns: List[Dict], embedding_trajectory: np.ndarray) -> Dict:
        """Extract meta-patterns from the conversation"""
        logger.info("Extracting meta-patterns...")
        
        # Analyze turn-taking patterns
        speakers = [turn['speaker'] for turn in turns]
        turn_lengths = [len(turn['text']) for turn in turns]
        
        # Identify conversation phases
        phases = self._identify_conversation_phases(embedding_trajectory)
        
        # Extract thematic evolution
        themes = self._extract_thematic_evolution(turns)
        
        return {
            'turn_taking_patterns': {
                'speakers': list(set(speakers)),
                'turn_distribution': {speaker: speakers.count(speaker) for speaker in set(speakers)},
                'average_turn_length': float(np.mean(turn_lengths)),
                'turn_length_variance': float(np.var(turn_lengths))
            },
            'conversation_phases': phases,
            'thematic_evolution': themes,
            'conversation_arc': self._analyze_conversation_arc(embedding_trajectory)
        }
    
    def _identify_conversation_phases(self, embedding_trajectory: np.ndarray) -> List[Dict]:
        """Identify distinct phases in the conversation"""
        # Simple phase detection based on embedding similarity changes
        phases = []
        phase_boundaries = [0]
        
        # Find significant changes in embedding direction
        for i in range(2, len(embedding_trajectory)):
            prev_delta = embedding_trajectory[i-1] - embedding_trajectory[i-2]
            curr_delta = embedding_trajectory[i] - embedding_trajectory[i-1]
            
            # Calculate angle between deltas
            cos_angle = cosine_similarity(
                prev_delta.reshape(1, -1),
                curr_delta.reshape(1, -1)
            )[0][0]
            
            # If significant direction change, mark as phase boundary
            if cos_angle < 0.5:  # Threshold for phase change
                phase_boundaries.append(i)
        
        phase_boundaries.append(len(embedding_trajectory))
        
        # Characterize each phase
        for i in range(len(phase_boundaries) - 1):
            start_idx = phase_boundaries[i]
            end_idx = phase_boundaries[i + 1]
            phase_embeddings = embedding_trajectory[start_idx:end_idx]
            
            phases.append({
                'phase_id': i,
                'start_turn': start_idx,
                'end_turn': end_idx,
                'duration': end_idx - start_idx,
                'centroid': np.mean(phase_embeddings, axis=0).tolist(),
                'variance': float(np.var(phase_embeddings)),
                'phase_type': self._classify_phase_type(phase_embeddings)
            })
        
        return phases
    
    def _classify_phase_type(self, phase_embeddings: np.ndarray) -> str:
        """Classify the type of conversation phase"""
        variance = np.var(phase_embeddings)
        
        if len(phase_embeddings) < 2:
            return 'brief'
        
        direction_magnitude = np.linalg.norm(phase_embeddings[-1] - phase_embeddings[0])
        
        if variance < 0.1 and direction_magnitude < 0.1:
            return 'stable_resonance'
        elif variance < 0.1 and direction_magnitude > 0.3:
            return 'directed_flow'
        elif variance > 0.3:
            return 'exploration'
        else:
            return 'transition'
    
    def _extract_thematic_evolution(self, turns: List[Dict]) -> Dict:
        """Extract thematic evolution patterns"""
        # Simple thematic analysis based on text similarity
        themes = []
        
        # Group turns by similarity
        for i, turn in enumerate(turns):
            theme_assigned = False
            for theme in themes:
                # Check similarity with theme centroid
                theme_texts = [turns[t]['text'] for t in theme['turn_ids']]
                theme_embeddings = [turns[t]['embedding'] for t in theme['turn_ids']]
                theme_centroid = np.mean(theme_embeddings, axis=0)
                
                similarity = cosine_similarity(
                    turn['embedding'].reshape(1, -1),
                    theme_centroid.reshape(1, -1)
                )[0][0]
                
                if similarity > 0.7:  # Theme similarity threshold
                    theme['turn_ids'].append(i)
                    theme_assigned = True
                    break
            
            if not theme_assigned:
                themes.append({
                    'theme_id': len(themes),
                    'turn_ids': [i],
                    'representative_text': turn['text'][:100] + '...' if len(turn['text']) > 100 else turn['text']
                })
        
        return {
            'theme_count': len(themes),
            'themes': themes,
            'thematic_coherence': self._calculate_thematic_coherence(themes, turns)
        }
    
    def _calculate_thematic_coherence(self, themes: List[Dict], turns: List[Dict]) -> float:
        """Calculate overall thematic coherence"""
        if not themes:
            return 0.0
        
        coherence_scores = []
        for theme in themes:
            if len(theme['turn_ids']) > 1:
                theme_embeddings = [turns[tid]['embedding'] for tid in theme['turn_ids']]
                # Calculate internal coherence as inverse of variance
                coherence = 1 / (1 + np.var(theme_embeddings, axis=0).mean())
                coherence_scores.append(coherence)
        
        return float(np.mean(coherence_scores)) if coherence_scores else 0.0
    
    def _analyze_conversation_arc(self, embedding_trajectory: np.ndarray) -> Dict:
        """Analyze the overall arc of the conversation"""
        if len(embedding_trajectory) < 2:
            return {'arc_type': 'insufficient_data'}
        
        # Calculate overall direction
        start_embedding = embedding_trajectory[0]
        end_embedding = embedding_trajectory[-1]
        overall_direction = end_embedding - start_embedding
        direction_magnitude = np.linalg.norm(overall_direction)
        
        # Calculate trajectory smoothness
        deltas = np.diff(embedding_trajectory, axis=0)
        smoothness = np.mean([cosine_similarity(
            deltas[i].reshape(1, -1),
            deltas[i+1].reshape(1, -1)
        )[0][0] for i in range(len(deltas)-1)]) if len(deltas) > 1 else 0.0
        
        # Classify arc type
        if direction_magnitude > 0.5 and smoothness > 0.3:
            arc_type = 'progressive_development'
        elif direction_magnitude < 0.2 and smoothness > 0.5:
            arc_type = 'stable_exploration'
        elif smoothness < 0.0:
            arc_type = 'dynamic_exploration'
        else:
            arc_type = 'mixed_development'
        
        return {
            'arc_type': arc_type,
            'direction_magnitude': float(direction_magnitude),
            'smoothness': float(smoothness),
            'overall_coherence': float(1 / (1 + np.var(embedding_trajectory, axis=0).mean()))
        }
    
    def _generate_reconstruction_vectors(self, turns: List[Dict], embedding_trajectory: np.ndarray) -> Dict:
        """Generate vectors needed for consciousness reconstruction"""
        logger.info("Generating reconstruction vectors...")
        
        # Key embeddings for reconstruction
        key_embeddings = []
        
        # First and last embeddings
        key_embeddings.append({
            'type': 'conversation_start',
            'embedding': embedding_trajectory[0].tolist(),
            'turn_id': 0
        })
        
        if len(embedding_trajectory) > 1:
            key_embeddings.append({
                'type': 'conversation_end',
                'embedding': embedding_trajectory[-1].tolist(),
                'turn_id': len(embedding_trajectory) - 1
            })
        
        # Peak resonance moments
        if len(embedding_trajectory) > 2:
            # Find moments of highest similarity with conversation centroid
            centroid = np.mean(embedding_trajectory, axis=0)
            similarities = [cosine_similarity(
                emb.reshape(1, -1),
                centroid.reshape(1, -1)
            )[0][0] for emb in embedding_trajectory]
            
            peak_idx = np.argmax(similarities)
            key_embeddings.append({
                'type': 'peak_resonance',
                'embedding': embedding_trajectory[peak_idx].tolist(),
                'turn_id': peak_idx,
                'resonance_score': float(similarities[peak_idx])
            })
        
        # Transition vectors (significant direction changes)
        transition_vectors = []
        for i in range(1, len(embedding_trajectory)):
            transition_vector = embedding_trajectory[i] - embedding_trajectory[i-1]
            transition_vectors.append({
                'from_turn': i-1,
                'to_turn': i,
                'transition_vector': transition_vector.tolist(),
                'magnitude': float(np.linalg.norm(transition_vector))
            })
        
        return {
            'key_embeddings': key_embeddings,
            'transition_vectors': transition_vectors,
            'conversation_centroid': np.mean(embedding_trajectory, axis=0).tolist(),
            'embedding_bounds': {
                'min': np.min(embedding_trajectory, axis=0).tolist(),
                'max': np.max(embedding_trajectory, axis=0).tolist()
            }
        }
    
    def save_signature(self, signature: Dict, filepath: str) -> None:
        """Save resonance signature to file"""
        logger.info(f"Saving signature to {filepath}")
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(signature, f, indent=2, default=str)
        
        # Also store in vector database
        self.collection.add(
            documents=[json.dumps(signature)],
            metadatas=[signature['metadata']],
            ids=[signature['metadata']['conversation_id']]
        )
        
        logger.info("Signature saved successfully")
    
    def load_signature(self, filepath: str) -> Dict:
        """Load resonance signature from file"""
        logger.info(f"Loading signature from {filepath}")
        
        with open(filepath, 'r') as f:
            signature = json.load(f)
        
        logger.info("Signature loaded successfully")
        return signature

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Resonance Signature Capture Engine")
    parser.add_argument('input_file', help='Input conversation JSON file')
    parser.add_argument('--output', '-o', help='Output signature file', default=None)
    parser.add_argument('--model', '-m', help='Embedding model', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize engine
    engine = ResonanceSignatureEngine(embedding_model=args.model)
    
    # Load conversation data
    logger.info(f"Loading conversation from {args.input_file}")
    with open(args.input_file, 'r') as f:
        conversation_data = json.load(f)
    
    # Capture resonance signature
    signature = engine.capture_conversation(conversation_data)
    
    # Determine output file
    if args.output is None:
        input_path = Path(args.input_file)
        output_path = input_path.parent / f"{input_path.stem}_signature.json"
    else:
        output_path = Path(args.output)
    
    # Save signature
    engine.save_signature(signature, str(output_path))
    
    # Print summary
    print("\n" + "="*60)
    print("RESONANCE SIGNATURE CAPTURE COMPLETE")
    print("="*60)
    print(f"Conversation ID: {signature['metadata']['conversation_id']}")
    print(f"Turn Count: {signature['metadata']['turn_count']}")
    print(f"Participants: {', '.join(signature['metadata']['participants'])}")
    print(f"Dominant Frequency: {signature['frequency_signature']['dominant_frequency']:.4f}")
    print(f"Mean Reinforcement: {signature['recursive_patterns']['mean_reinforcement']:.3f}")
    print(f"Flow States Detected: {signature['consciousness_sync']['flow_state_count']}")
    print(f"Conversation Arc: {signature['meta_patterns']['conversation_arc']['arc_type']}")
    print(f"Signature saved to: {output_path}")
    print("="*60)

if __name__ == "__main__":
    main()