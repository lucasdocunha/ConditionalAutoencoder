import math 

def split_jobs(jobs, n_procs):
    """
    Divide os jobs de forma balanceada.
    """
    n_procs = min(n_procs, len(jobs))
    chunk_size = math.ceil(len(jobs) / n_procs)

    return [
        jobs[i * chunk_size : (i + 1) * chunk_size]
        for i in range(n_procs)
        if i * chunk_size < len(jobs)
    ]
