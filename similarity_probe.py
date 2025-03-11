import torch
import torch.nn.functional as F
import numpy as np

def compute_kl_divergence(p, q):
    # p, q are probability distributions (torch tensors), shape [384]
    # Return scalar KL-Divergence D_{KL}(p || q)
    kl_div = (p * (p.log() - q.log())).sum()
    return kl_div

def similarity_probe(tensor, k=0.1, output_file='cushionCache.txt'):
    # tensor: shape [256,196,384]
    # Open the output file in append mode
    with open(output_file, 'a') as f_out:
        for sample_idx in range(tensor.shape[0]):
            sample = tensor[sample_idx]  # shape [196,384]

            epsilon = 1e-10  # Small value to avoid log(0)

            # Normalize the 384-dimensional vectors to probability distributions
            distributions = F.softmax(sample, dim=1) + epsilon  # Apply softmax along the 384 dimension
            distributions = distributions / distributions.sum(dim=1, keepdim=True)

            clusters = []  # List of clusters

            for dist_idx in range(distributions.shape[0]):
                dist = distributions[dist_idx]  # shape [384]

                assigned = False

                for cluster in clusters:
                    center = cluster['center']  # shape [384]
                    kl_div = compute_kl_divergence(dist, center)

                    if kl_div <= k:
                        # Add dist to cluster
                        cluster['sum'] += dist
                        cluster['count'] += 1
                        # Update center
                        cluster['center'] = cluster['sum'] / cluster['count']
                        cluster['center'] = cluster['center'] / cluster['center'].sum()  # Normalize
                        cluster['members'].append(dist)
                        assigned = True
                        break

                if not assigned:
                    # Create new cluster
                    clusters.append({
                        'center': dist,
                        'sum': dist.clone(),  # make a copy
                        'count': 1,
                        'members': [dist],
                    })

            # After clustering, rank clusters by number of elements
            clusters = sorted(clusters, key=lambda x: x['count'], reverse=True)

            # Get the top cluster
            top_cluster = clusters[0]

            # Compute mean and std of distances between elements and cluster center
            distances = []
            center = top_cluster['center']

            for member in top_cluster['members']:
                kl_div = compute_kl_divergence(member, center)
                distances.append(kl_div.item())

            distances = np.array(distances)

            mean_distance = distances.mean()
            std_distance = distances.std()

            # Output the results
            f_out.write(f"Sample {sample_idx}\n")
            f_out.write(f"Top cluster has {len(top_cluster['members'])} elements\n")
            f_out.write(f"Mean KL-Divergence to center: {mean_distance}\n")
            f_out.write(f"Standard deviation of KL-Divergence: {std_distance}\n")
            f_out.write(f"Cluster center vector:\n")
            np.set_printoptions(suppress=True)
            f_out.write(f"{top_cluster['center'].cpu().numpy()}\n")
            f_out.write("\n")
