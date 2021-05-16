import numpy as np

def knn(x_train, y_train, x_test, k):
    y_pred = np.zeros(x_test.shape[0], dtype=np.int)
    for i in range(x_test.shape[0]):
        dist = np.sqrt(((x_train - x_test[i]) ** 2).sum(axis=1))
        dist_sorted = np.sort(dist)
        kk = k
        while kk < dist.shape[0] and dist_sorted[k-1] == dist_sorted[kk]:
            kk += 1
        indices = np.argsort(dist)[:kk]
        bins = np.zeros(y_train.max() + 1, dtype=np.int)
        for ind in indices:
            bins[y_train[ind]] += 1
        y_pred[i] = np.argmax(bins)
    return y_pred

def knn_kdt(x_train, y_train, x_test, k):
    y_pred = np.zeros(x_test.shape[0], dtype=np.int)
    kdt = KDTree()
    kdt.create_kdtree(x_train)
    for i in range(x_test.shape[0]):
        indices = kdt.search_knn(x_test[i], k)
        bins = np.zeros(y_train.max() + 1, dtype=np.int)
        for ind in indices:
            bins[y_train[ind]] += 1
        y_pred[i] = np.argmax(bins)
    return y_pred


class KDTree(object):
    '''k-d树的实现类

    k-d树搜索函数参考：
    https://github.com/heshenghuan/python-KNN/blob/master/kdtree.py
    '''
    class Node(object):
        def __init__(self, data, dim):
            self.data = data
            self.dim = dim
            self.parent = None
            self.left = None
            self.right = None

    def __init__(self):
        self.root = None
        self.x = None

    def create_kdtree(self, x):
        self.x = list(x)
        # 因为建树过程中训练集需要重新排序，每个样本后添加初始索引
        for i in range(len(x)):
            self.x[i] = np.append(x[i], i)
        self._split(None, 0, len(x), 0)

    def _split(self, parent, start, end, dim, is_left=True):
        if start >= end:
            return
        self.x[start:end] = sorted(self.x[start:end], key=lambda x: x[dim])
        split = start + (end - start) // 2
        if parent is None:
            self.root = KDTree.Node(self.x[split], dim)
            this = self.root
        elif is_left:
            parent.left = KDTree.Node(self.x[split], dim)
            parent.left.parent = parent
            this = parent.left
        elif not is_left:
            parent.right = KDTree.Node(self.x[split], dim)
            parent.right.parent = parent
            this = parent.right
        dim = (dim + 1) % (len(self.x[0]) - 1)
        self._split(this, start, split, dim, True)
        self._split(this, split + 1, end, dim, False)

    def preorder(self, root=None):
        if root == None:
            self.preorder(self.root)
            return
        if root.data is not None:
            print(root.data)
        if root.left is not None:
            self.preorder(root.left)
        if root.right is not None:
            self.preorder(root.right)

    def search_knn(self, x, k):
        dim = 0
        cur = self.root
        # 找到所属最小区块的节点
        while cur is not None:
            prev = cur
            if x[dim] < cur.data[dim]:
                cur = cur.left
            else:
                cur = cur.right
            dim = (dim + 1) % (len(self.x[0]) - 1)

        cur = prev
        examined = set()
        results = {}
        while cur:
            self._search(cur, x, k, examined, results)
            cur = cur.parent

        return [int(node.data[-1]) for node in results.keys()]

    def _search(self, p, x, k, examined, results):
        examined.add(p)
        if not results:
            bestDist = np.inf
        else:
            bestDist = sorted(results.values())[0]

        # 根据目标点和节点的距离判断是否要加入最近邻集合
        no_change = False
        dist = np.sqrt(sum((p.data[:-1] - np.array(x)) ** 2))
        if dist < bestDist:
            if len(results) == k:
                maxNode, maxDist = sorted(results.items(), key=lambda x:x[1], reverse=True)[0]
                results.pop(maxNode)
            results[p] = dist
        elif dist == bestDist or len(results) < k:
            results[p] = dist
        else:
            no_change = True

        if not no_change:
            bestDist = sorted(results.values())[0]

        for child in [p.left, p.right]:
            if child in examined or child is None:
                continue
            examined.add(child)

            # 根据目标点和划分超平面的距离剪枝
            if abs(p.data[p.dim] - x[p.dim]) < bestDist or len(results) < k:
                self._search(child, x, k, examined, results)

if __name__ == '__main__':
    kd=KDTree()
    kd.create_kdtree([[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]])
    kd.preorder()
    kd.search_knn((1,2),5)
