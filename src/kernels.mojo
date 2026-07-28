"""Sparse graph kernels exposed through a stable C ABI."""

from std.math import abs, sqrt
from std.sys.info import simd_width_of

comptime W = simd_width_of[DType.float64]()
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime INF = 1.7976931348623157e308


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def heap_swap(heap: IPtr, pos: IPtr, a: Int, b: Int):
    var x = heap[a]
    var y = heap[b]
    heap[a] = y
    heap[b] = x
    pos[Int(x)] = Int64(b)
    pos[Int(y)] = Int64(a)


def heap_up(heap: IPtr, pos: IPtr, dist: FPtr, start: Int):
    var child = start
    while child > 0:
        var parent = (child - 1) // 2
        if dist[Int(heap[parent])] <= dist[Int(heap[child])]:
            break
        heap_swap(heap, pos, parent, child)
        child = parent


def heap_down(heap: IPtr, pos: IPtr, dist: FPtr, size: Int, start: Int):
    var parent = start
    while True:
        var left = parent * 2 + 1
        if left >= size:
            break
        var best = left
        var right = left + 1
        if right < size and dist[Int(heap[right])] < dist[Int(heap[left])]:
            best = right
        if dist[Int(heap[parent])] <= dist[Int(heap[best])]:
            break
        heap_swap(heap, pos, parent, best)
        parent = best


def dijkstra_impl(
    indptr: IPtr,
    indices: IPtr,
    weights: FPtr,
    n: Int,
    source: Int,
    cutoff: Float64,
    dist: FPtr,
    pred: IPtr,
    heap: IPtr,
    pos: IPtr,
    order: IPtr,
) -> Int:
    for i in range(n):
        dist[i] = INF
        pred[i] = -1
        pos[i] = -1
    dist[source] = 0.0
    heap[0] = Int64(source)
    pos[source] = 0
    var size = 1
    var count = 0
    while size > 0:
        var v = Int(heap[0])
        size -= 1
        pos[v] = -2
        if size > 0:
            heap[0] = heap[size]
            pos[Int(heap[0])] = 0
            heap_down(heap, pos, dist, size, 0)
        if dist[v] > cutoff:
            break
        order[count] = Int64(v)
        count += 1
        for edge in range(Int(indptr[v]), Int(indptr[v + 1])):
            var w = Int(indices[edge])
            if pos[w] == -2:
                continue
            var candidate = dist[v] + weights[edge]
            if candidate > cutoff or candidate >= dist[w]:
                continue
            dist[w] = candidate
            pred[w] = Int64(v)
            if pos[w] == -1:
                heap[size] = Int64(w)
                pos[w] = Int64(size)
                heap_up(heap, pos, dist, size)
                size += 1
            else:
                heap_up(heap, pos, dist, Int(pos[w]))
    return count


@export("mnx_bfs")
def mnx_bfs(
    indptr_addr: Int,
    indices_addr: Int,
    n: Int,
    source: Int,
    depth_limit: Int,
    queue_addr: Int,
    dist_addr: Int,
    pred_addr: Int,
    order_addr: Int,
) abi("C") -> Int:
    var indptr = ip(indptr_addr)
    var indices = ip(indices_addr)
    var queue = ip(queue_addr)
    var dist = ip(dist_addr)
    var pred = ip(pred_addr)
    var order = ip(order_addr)
    for i in range(n):
        dist[i] = -1
        pred[i] = -1
    dist[source] = 0
    queue[0] = Int64(source)
    var head = 0
    var tail = 1
    while head < tail:
        var v = Int(queue[head])
        head += 1
        order[head - 1] = Int64(v)
        if Int(dist[v]) >= depth_limit:
            continue
        for edge in range(Int(indptr[v]), Int(indptr[v + 1])):
            var w = Int(indices[edge])
            if dist[w] < 0:
                dist[w] = dist[v] + 1
                pred[w] = Int64(v)
                queue[tail] = Int64(w)
                tail += 1
    return tail


@export("mnx_dfs")
def mnx_dfs(
    indptr_addr: Int,
    indices_addr: Int,
    n: Int,
    source: Int,
    depth_limit: Int,
    seen_addr: Int,
    stack_node_addr: Int,
    stack_edge_addr: Int,
    depth_addr: Int,
    pred_addr: Int,
    preorder_addr: Int,
    postorder_addr: Int,
) abi("C") -> Int:
    var indptr = ip(indptr_addr)
    var indices = ip(indices_addr)
    var seen = ip(seen_addr)
    var stack_node = ip(stack_node_addr)
    var stack_edge = ip(stack_edge_addr)
    var depth = ip(depth_addr)
    var pred = ip(pred_addr)
    var preorder = ip(preorder_addr)
    var postorder = ip(postorder_addr)
    for i in range(n):
        seen[i] = 0
        pred[i] = -1
    var pre_count = 0
    var post_count = 0
    var cursor = source if source >= 0 else 0
    while cursor < n:
        if seen[cursor] != 0:
            cursor += 1
            continue
        seen[cursor] = 1
        stack_node[0] = Int64(cursor)
        stack_edge[0] = indptr[cursor]
        depth[0] = 1
        preorder[pre_count] = Int64(cursor)
        pre_count += 1
        var top = 0
        while top >= 0:
            var v = Int(stack_node[top])
            if stack_edge[top] >= indptr[v + 1]:
                postorder[post_count] = Int64(v)
                post_count += 1
                top -= 1
                continue
            var edge = Int(stack_edge[top])
            stack_edge[top] += 1
            var w = Int(indices[edge])
            if seen[w] != 0:
                continue
            seen[w] = 1
            pred[w] = Int64(v)
            preorder[pre_count] = Int64(w)
            pre_count += 1
            if Int(depth[top]) >= depth_limit:
                continue
            top += 1
            stack_node[top] = Int64(w)
            stack_edge[top] = indptr[w]
            depth[top] = depth[top - 1] + 1
        if source >= 0:
            break
        cursor += 1
    depth[0] = Int64(post_count)
    return pre_count


@export("mnx_components")
def mnx_components(
    indptr_addr: Int,
    indices_addr: Int,
    n: Int,
    labels_addr: Int,
    queue_addr: Int,
) abi("C") -> Int:
    var indptr = ip(indptr_addr)
    var indices = ip(indices_addr)
    var labels = ip(labels_addr)
    var queue = ip(queue_addr)
    for i in range(n):
        labels[i] = -1
    var component = 0
    for source in range(n):
        if labels[source] >= 0:
            continue
        labels[source] = Int64(component)
        queue[0] = Int64(source)
        var head = 0
        var tail = 1
        while head < tail:
            var v = Int(queue[head])
            head += 1
            for edge in range(Int(indptr[v]), Int(indptr[v + 1])):
                var w = Int(indices[edge])
                if labels[w] < 0:
                    labels[w] = Int64(component)
                    queue[tail] = Int64(w)
                    tail += 1
        component += 1
    return component


@export("mnx_scc")
def mnx_scc(
    indptr_addr: Int,
    indices_addr: Int,
    revptr_addr: Int,
    revindices_addr: Int,
    n: Int,
    labels_addr: Int,
    seen_addr: Int,
    order_addr: Int,
    stack_node_addr: Int,
    stack_edge_addr: Int,
) abi("C") -> Int:
    var indptr = ip(indptr_addr)
    var indices = ip(indices_addr)
    var revptr = ip(revptr_addr)
    var revindices = ip(revindices_addr)
    var labels = ip(labels_addr)
    var seen = ip(seen_addr)
    var order = ip(order_addr)
    var stack_node = ip(stack_node_addr)
    var stack_edge = ip(stack_edge_addr)
    for i in range(n):
        seen[i] = 0
        labels[i] = -1
    var finished = 0
    for source in range(n):
        if seen[source] != 0:
            continue
        seen[source] = 1
        stack_node[0] = Int64(source)
        stack_edge[0] = indptr[source]
        var top = 0
        while top >= 0:
            var v = Int(stack_node[top])
            if stack_edge[top] >= indptr[v + 1]:
                order[finished] = Int64(v)
                finished += 1
                top -= 1
                continue
            var edge = Int(stack_edge[top])
            stack_edge[top] += 1
            var w = Int(indices[edge])
            if seen[w] == 0:
                seen[w] = 1
                top += 1
                stack_node[top] = Int64(w)
                stack_edge[top] = indptr[w]
    var component = 0
    for offset in range(n):
        var source = Int(order[n - 1 - offset])
        if labels[source] >= 0:
            continue
        labels[source] = Int64(component)
        stack_node[0] = Int64(source)
        var size = 1
        while size > 0:
            size -= 1
            var v = Int(stack_node[size])
            for edge in range(Int(revptr[v]), Int(revptr[v + 1])):
                var w = Int(revindices[edge])
                if labels[w] < 0:
                    labels[w] = Int64(component)
                    stack_node[size] = Int64(w)
                    size += 1
        component += 1
    return component


@export("mnx_dijkstra")
def mnx_dijkstra(
    indptr_addr: Int,
    indices_addr: Int,
    weights_addr: Int,
    n: Int,
    source: Int,
    cutoff: Float64,
    dist_addr: Int,
    pred_addr: Int,
    heap_addr: Int,
    pos_addr: Int,
    order_addr: Int,
) abi("C") -> Int:
    return dijkstra_impl(
        ip(indptr_addr), ip(indices_addr), fp(weights_addr), n, source, cutoff,
        fp(dist_addr), ip(pred_addr), ip(heap_addr), ip(pos_addr), ip(order_addr),
    )


@export("mnx_betweenness")
def mnx_betweenness(
    indptr_addr: Int,
    indices_addr: Int,
    n: Int,
    endpoints: Int,
    centrality_addr: Int,
    dist_addr: Int,
    sigma_addr: Int,
    delta_addr: Int,
    queue_addr: Int,
    order_addr: Int,
) abi("C"):
    var indptr = ip(indptr_addr)
    var indices = ip(indices_addr)
    var centrality = fp(centrality_addr)
    var dist = ip(dist_addr)
    var sigma = fp(sigma_addr)
    var delta = fp(delta_addr)
    var queue = ip(queue_addr)
    var order = ip(order_addr)
    for i in range(n):
        centrality[i] = 0.0
    for source in range(n):
        for i in range(n):
            dist[i] = -1
            sigma[i] = 0.0
            delta[i] = 0.0
        dist[source] = 0
        sigma[source] = 1.0
        queue[0] = Int64(source)
        var head = 0
        var tail = 1
        while head < tail:
            var v = Int(queue[head])
            order[head] = Int64(v)
            head += 1
            for edge in range(Int(indptr[v]), Int(indptr[v + 1])):
                var w = Int(indices[edge])
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue[tail] = Int64(w)
                    tail += 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
        for offset in range(tail):
            var v = Int(order[tail - 1 - offset])
            for edge in range(Int(indptr[v]), Int(indptr[v + 1])):
                var w = Int(indices[edge])
                if dist[w] == dist[v] + 1:
                    delta[v] += sigma[v] / sigma[w] * (1.0 + delta[w])
            if v != source:
                centrality[v] += delta[v] + (1.0 if endpoints != 0 else 0.0)
        if endpoints != 0:
            centrality[source] += Float64(tail - 1)


@export("mnx_betweenness_weighted")
def mnx_betweenness_weighted(
    indptr_addr: Int,
    indices_addr: Int,
    weights_addr: Int,
    n: Int,
    endpoints: Int,
    centrality_addr: Int,
    dist_addr: Int,
    sigma_addr: Int,
    delta_addr: Int,
    heap_addr: Int,
    pos_addr: Int,
    order_addr: Int,
) abi("C"):
    var indptr = ip(indptr_addr)
    var indices = ip(indices_addr)
    var weights = fp(weights_addr)
    var centrality = fp(centrality_addr)
    var dist = fp(dist_addr)
    var sigma = fp(sigma_addr)
    var delta = fp(delta_addr)
    var heap = ip(heap_addr)
    var pos = ip(pos_addr)
    var order = ip(order_addr)
    for i in range(n):
        centrality[i] = 0.0
    for source in range(n):
        for i in range(n):
            dist[i] = INF
            sigma[i] = 0.0
            delta[i] = 0.0
            pos[i] = -1
        dist[source] = 0.0
        sigma[source] = 1.0
        heap[0] = Int64(source)
        pos[source] = 0
        var size = 1
        var count = 0
        while size > 0:
            var v = Int(heap[0])
            size -= 1
            pos[v] = -2
            if size > 0:
                heap[0] = heap[size]
                pos[Int(heap[0])] = 0
                heap_down(heap, pos, dist, size, 0)
            order[count] = Int64(v)
            count += 1
            for edge in range(Int(indptr[v]), Int(indptr[v + 1])):
                var w = Int(indices[edge])
                if pos[w] == -2:
                    continue
                var candidate = dist[v] + weights[edge]
                if candidate < dist[w]:
                    dist[w] = candidate
                    sigma[w] = sigma[v]
                    if pos[w] == -1:
                        heap[size] = Int64(w)
                        pos[w] = Int64(size)
                        heap_up(heap, pos, dist, size)
                        size += 1
                    else:
                        heap_up(heap, pos, dist, Int(pos[w]))
                elif candidate == dist[w]:
                    sigma[w] += sigma[v]
        for offset in range(count):
            var v = Int(order[count - 1 - offset])
            for edge in range(Int(indptr[v]), Int(indptr[v + 1])):
                var w = Int(indices[edge])
                if dist[w] < INF and dist[v] + weights[edge] == dist[w]:
                    delta[v] += sigma[v] / sigma[w] * (1.0 + delta[w])
            if v != source:
                centrality[v] += delta[v] + (1.0 if endpoints != 0 else 0.0)
        if endpoints != 0:
            centrality[source] += Float64(count - 1)


@export("mnx_closeness_unweighted")
def mnx_closeness_unweighted(
    indptr_addr: Int,
    indices_addr: Int,
    n: Int,
    wf_improved: Int,
    scores_addr: Int,
    queue_addr: Int,
    dist_addr: Int,
) abi("C"):
    var indptr = ip(indptr_addr)
    var indices = ip(indices_addr)
    var scores = fp(scores_addr)
    var queue = ip(queue_addr)
    var dist = ip(dist_addr)
    for source in range(n):
        for i in range(n):
            dist[i] = -1
        dist[source] = 0
        queue[0] = Int64(source)
        var head = 0
        var tail = 1
        var total = 0.0
        while head < tail:
            var v = Int(queue[head])
            head += 1
            for edge in range(Int(indptr[v]), Int(indptr[v + 1])):
                var w = Int(indices[edge])
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    total += Float64(dist[w])
                    queue[tail] = Int64(w)
                    tail += 1
        if total > 0.0 and n > 1:
            scores[source] = Float64(tail - 1) / total
            if wf_improved != 0:
                scores[source] *= Float64(tail - 1) / Float64(n - 1)
        else:
            scores[source] = 0.0


@export("mnx_closeness_weighted")
def mnx_closeness_weighted(
    indptr_addr: Int,
    indices_addr: Int,
    weights_addr: Int,
    n: Int,
    wf_improved: Int,
    scores_addr: Int,
    dist_addr: Int,
    pred_addr: Int,
    heap_addr: Int,
    pos_addr: Int,
    order_addr: Int,
) abi("C"):
    var scores = fp(scores_addr)
    var dist = fp(dist_addr)
    var order = ip(order_addr)
    for source in range(n):
        var count = dijkstra_impl(
            ip(indptr_addr), ip(indices_addr), fp(weights_addr), n, source, INF,
            dist, ip(pred_addr), ip(heap_addr), ip(pos_addr), order,
        )
        var total = 0.0
        for j in range(count):
            total += dist[Int(order[j])]
        if total > 0.0 and n > 1:
            scores[source] = Float64(count - 1) / total
            if wf_improved != 0:
                scores[source] *= Float64(count - 1) / Float64(n - 1)
        else:
            scores[source] = 0.0


@export("mnx_pagerank")
def mnx_pagerank(
    indptr_addr: Int,
    indices_addr: Int,
    weights_addr: Int,
    n: Int,
    alpha: Float64,
    max_iter: Int,
    tol: Float64,
    rank_addr: Int,
    next_addr: Int,
    strength_addr: Int,
) abi("C") -> Int:
    var indptr = ip(indptr_addr)
    var indices = ip(indices_addr)
    var weights = fp(weights_addr)
    var rank = fp(rank_addr)
    var next = fp(next_addr)
    var strength = fp(strength_addr)
    var uniform = 1.0 / Float64(n)
    var i = 0
    while i + W <= n:
        rank.store(i, SIMD[DType.float64, W](uniform))
        i += W
    while i < n:
        rank[i] = uniform
        i += 1

    for v in range(n):
        strength[v] = 0.0
        for edge in range(Int(indptr[v]), Int(indptr[v + 1])):
            strength[v] += weights[edge]

    for iteration in range(max_iter):
        var dangling = 0.0
        for v in range(n):
            if strength[v] == 0.0:
                dangling += rank[v]
        var base = (1.0 - alpha + alpha * dangling) / Float64(n)
        i = 0
        while i + W <= n:
            next.store(i, SIMD[DType.float64, W](base))
            i += W
        while i < n:
            next[i] = base
            i += 1
        for v in range(n):
            if strength[v] == 0.0:
                continue
            var scale = alpha * rank[v] / strength[v]
            for edge in range(Int(indptr[v]), Int(indptr[v + 1])):
                next[Int(indices[edge])] += scale * weights[edge]
        var error_vec = SIMD[DType.float64, W](0.0)
        i = 0
        while i + W <= n:
            var next_vec = next.load[width=W](i)
            error_vec += abs(next_vec - rank.load[width=W](i))
            rank.store(i, next_vec)
            i += W
        var error = error_vec.reduce_add()
        while i < n:
            error += abs(next[i] - rank[i])
            rank[i] = next[i]
            i += 1
        if error < Float64(n) * tol:
            return iteration + 1
    return -1


@export("mnx_eigenvector")
def mnx_eigenvector(
    indptr_addr: Int,
    indices_addr: Int,
    weights_addr: Int,
    n: Int,
    max_iter: Int,
    tol: Float64,
    vector_addr: Int,
    next_addr: Int,
) abi("C") -> Int:
    var indptr = ip(indptr_addr)
    var indices = ip(indices_addr)
    var weights = fp(weights_addr)
    var vector = fp(vector_addr)
    var next = fp(next_addr)
    for v in range(n):
        vector[v] = 1.0 / Float64(n)
    for iteration in range(max_iter):
        for v in range(n):
            next[v] = vector[v]
        for v in range(n):
            for edge in range(Int(indptr[v]), Int(indptr[v + 1])):
                next[Int(indices[edge])] += vector[v] * weights[edge]
        var norm2 = 0.0
        for v in range(n):
            norm2 += next[v] * next[v]
        var norm = sqrt(norm2)
        if norm == 0.0:
            norm = 1.0
        var error = 0.0
        for v in range(n):
            next[v] /= norm
            error += abs(next[v] - vector[v])
            vector[v] = next[v]
        if error < Float64(n) * tol:
            return iteration + 1
    return -1
