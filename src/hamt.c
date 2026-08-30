#include "pds_internal.h"

// === Map Nodes ===

// Forward declarations for iterator functions
PyObject *BitmapIndexedNode_iter_mode(BitmapIndexedNode *self, int mode);
PyObject *ArrayNode_iter_mode(ArrayNode *self, int mode);
PyObject *HashCollisionNode_iter_mode(HashCollisionNode *self, int mode);

// BitmapIndexedNode
typedef struct BitmapIndexedNode {
    PyObject_HEAD
    unsigned int bitmap;
    PyObject *array;  // list
    PyObject *transient_id;
} BitmapIndexedNode;

PyTypeObject BitmapIndexedNodeType;
BitmapIndexedNode *EMPTY_BIN = NULL;

static int BitmapIndexedNode_traverse(
    BitmapIndexedNode *self, visitproc visit, void *arg) {
    Py_VISIT(self->array);
    Py_VISIT(self->transient_id);
    return 0;
}

static int BitmapIndexedNode_clear(BitmapIndexedNode *self) {
    Py_CLEAR(self->array);
    Py_CLEAR(self->transient_id);
    return 0;
}

static void BitmapIndexedNode_dealloc(BitmapIndexedNode *self) {
    PyObject_GC_UnTrack(self);
    BitmapIndexedNode_clear(self);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

BitmapIndexedNode *BitmapIndexedNode_create(unsigned int bitmap, PyObject *array, PyObject *transient_id) {
    BitmapIndexedNode *node =
        (BitmapIndexedNode *)BitmapIndexedNodeType.tp_alloc(
            &BitmapIndexedNodeType, 0);
    if (!node) return NULL;

    node->bitmap = bitmap;
    node->array = array ? array : PyList_New(0);
    if (!node->array) {
        Py_DECREF(node);
        return NULL;
    }
    if (array) Py_INCREF(array);
    node->transient_id = transient_id;
    Py_XINCREF(transient_id);

    return node;
}

static int BitmapIndexedNode_is_editable(BitmapIndexedNode *self, PyObject *transient_id) {
    return transient_id != NULL && self->transient_id == transient_id;
}

BitmapIndexedNode *BitmapIndexedNode_ensure_editable(BitmapIndexedNode *self, PyObject *transient_id) {
    if (BitmapIndexedNode_is_editable(self, transient_id)) {
        Py_INCREF(self);
        return self;
    }
    PyObject *new_array = PyList_GetSlice(self->array, 0, PyList_Size(self->array));
    if (!new_array) return NULL;
    BitmapIndexedNode *result = BitmapIndexedNode_create(self->bitmap, new_array, transient_id);
    Py_DECREF(new_array);
    return result;
}

// Forward declarations for node operations
PyObject *BitmapIndexedNode_assoc(BitmapIndexedNode *self, int shift, Py_hash_t hash_val, PyObject *key, PyObject *val, PyObject *added_leaf, PyObject *transient_id);
PyObject *BitmapIndexedNode_find(BitmapIndexedNode *self, int shift, Py_hash_t hash_val, PyObject *key, PyObject *not_found);
PyObject *BitmapIndexedNode_dissoc(BitmapIndexedNode *self, int shift, Py_hash_t hash_val, PyObject *key, PyObject *removed_leaf, PyObject *transient_id);
PyObject *BitmapIndexedNode_iter_kv(BitmapIndexedNode *self);

// ArrayNode
typedef struct ArrayNode {
    PyObject_HEAD
    int count;
    PyObject *array;  // list of WIDTH nodes
    PyObject *transient_id;
} ArrayNode;

PyTypeObject ArrayNodeType;

static int ArrayNode_traverse(ArrayNode *self, visitproc visit, void *arg) {
    Py_VISIT(self->array);
    Py_VISIT(self->transient_id);
    return 0;
}

static int ArrayNode_clear(ArrayNode *self) {
    Py_CLEAR(self->array);
    Py_CLEAR(self->transient_id);
    return 0;
}

static void ArrayNode_dealloc(ArrayNode *self) {
    PyObject_GC_UnTrack(self);
    ArrayNode_clear(self);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static ArrayNode *ArrayNode_create(int count, PyObject *array, PyObject *transient_id) {
    ArrayNode *node = (ArrayNode *)ArrayNodeType.tp_alloc(&ArrayNodeType, 0);
    if (!node) return NULL;

    node->count = count;
    if (array) {
        node->array = array;
        Py_INCREF(array);
    } else {
        node->array = PyList_New(WIDTH);
        if (!node->array) {
            Py_DECREF(node);
            return NULL;
        }
        for (int i = 0; i < WIDTH; i++) {
            Py_INCREF(Py_None);
            PyList_SET_ITEM(node->array, i, Py_None);
        }
    }
    node->transient_id = transient_id;
    Py_XINCREF(transient_id);

    return node;
}

static int ArrayNode_is_editable(ArrayNode *self, PyObject *transient_id) {
    return transient_id != NULL && self->transient_id == transient_id;
}

ArrayNode *ArrayNode_ensure_editable(ArrayNode *self, PyObject *transient_id) {
    if (ArrayNode_is_editable(self, transient_id)) {
        Py_INCREF(self);
        return self;
    }
    PyObject *new_array = PyList_GetSlice(self->array, 0, WIDTH);
    if (!new_array) return NULL;
    ArrayNode *result = ArrayNode_create(self->count, new_array, transient_id);
    Py_DECREF(new_array);
    return result;
}

PyObject *ArrayNode_assoc(ArrayNode *self, int shift, Py_hash_t hash_val, PyObject *key, PyObject *val, PyObject *added_leaf, PyObject *transient_id);
PyObject *ArrayNode_find(ArrayNode *self, int shift, Py_hash_t hash_val, PyObject *key, PyObject *not_found);
PyObject *ArrayNode_dissoc(ArrayNode *self, int shift, Py_hash_t hash_val, PyObject *key, PyObject *removed_leaf, PyObject *transient_id);
PyObject *ArrayNode_iter_kv(ArrayNode *self);

// HashCollisionNode
typedef struct HashCollisionNode {
    PyObject_HEAD
    Py_hash_t hash;
    int count;
    PyObject *array;  // list [k1, v1, k2, v2, ...]
    PyObject *transient_id;
} HashCollisionNode;

PyTypeObject HashCollisionNodeType;

static int HashCollisionNode_traverse(
    HashCollisionNode *self, visitproc visit, void *arg) {
    Py_VISIT(self->array);
    Py_VISIT(self->transient_id);
    return 0;
}

static int HashCollisionNode_clear(HashCollisionNode *self) {
    Py_CLEAR(self->array);
    Py_CLEAR(self->transient_id);
    return 0;
}

static void HashCollisionNode_dealloc(HashCollisionNode *self) {
    PyObject_GC_UnTrack(self);
    HashCollisionNode_clear(self);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static HashCollisionNode *HashCollisionNode_create(Py_hash_t hash_val, int count, PyObject *array, PyObject *transient_id) {
    HashCollisionNode *node =
        (HashCollisionNode *)HashCollisionNodeType.tp_alloc(
            &HashCollisionNodeType, 0);
    if (!node) return NULL;

    node->hash = hash_val;
    node->count = count;
    node->array = array ? array : PyList_New(0);
    if (!node->array) {
        Py_DECREF(node);
        return NULL;
    }
    if (array) Py_INCREF(array);
    node->transient_id = transient_id;
    Py_XINCREF(transient_id);

    return node;
}

static int HashCollisionNode_is_editable(HashCollisionNode *self, PyObject *transient_id) {
    return transient_id != NULL && self->transient_id == transient_id;
}

HashCollisionNode *HashCollisionNode_ensure_editable(HashCollisionNode *self, PyObject *transient_id) {
    if (HashCollisionNode_is_editable(self, transient_id)) {
        Py_INCREF(self);
        return self;
    }
    PyObject *new_array = PyList_GetSlice(self->array, 0, PyList_Size(self->array));
    if (!new_array) return NULL;
    HashCollisionNode *result = HashCollisionNode_create(self->hash, self->count, new_array, transient_id);
    Py_DECREF(new_array);
    return result;
}

static int HashCollisionNode_find_index(HashCollisionNode *self, PyObject *key) {
    for (int i = 0; i < 2 * self->count; i += 2) {
        PyObject *k = PyList_GET_ITEM(self->array, i);
        int eq = PyObject_RichCompareBool(k, key, Py_EQ);
        if (eq < 0) return -2;  // Error
        if (eq) return i;
    }
    return -1;
}

PyObject *HashCollisionNode_assoc(HashCollisionNode *self, int shift, Py_hash_t hash_val, PyObject *key, PyObject *val, PyObject *added_leaf, PyObject *transient_id);
PyObject *HashCollisionNode_find(HashCollisionNode *self, int shift, Py_hash_t hash_val, PyObject *key, PyObject *not_found);
PyObject *HashCollisionNode_dissoc(HashCollisionNode *self, int shift, Py_hash_t hash_val, PyObject *key, PyObject *removed_leaf, PyObject *transient_id);
PyObject *HashCollisionNode_iter_kv(HashCollisionNode *self);

// Node type definitions
PyTypeObject BitmapIndexedNodeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.BitmapIndexedNode",
    .tp_basicsize = sizeof(BitmapIndexedNode),
    .tp_dealloc = (destructor)BitmapIndexedNode_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .tp_traverse = (traverseproc)BitmapIndexedNode_traverse,
    .tp_clear = (inquiry)BitmapIndexedNode_clear,
    .tp_alloc = PyType_GenericAlloc,
    .tp_free = PyObject_GC_Del,
};

PyTypeObject ArrayNodeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.ArrayNode",
    .tp_basicsize = sizeof(ArrayNode),
    .tp_dealloc = (destructor)ArrayNode_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .tp_traverse = (traverseproc)ArrayNode_traverse,
    .tp_clear = (inquiry)ArrayNode_clear,
    .tp_alloc = PyType_GenericAlloc,
    .tp_free = PyObject_GC_Del,
};

PyTypeObject HashCollisionNodeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.HashCollisionNode",
    .tp_basicsize = sizeof(HashCollisionNode),
    .tp_dealloc = (destructor)HashCollisionNode_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .tp_traverse = (traverseproc)HashCollisionNode_traverse,
    .tp_clear = (inquiry)HashCollisionNode_clear,
    .tp_alloc = PyType_GenericAlloc,
    .tp_free = PyObject_GC_Del,
};

// Helper to create a node for two key-value pairs
static PyObject *create_node(int shift, PyObject *key1, PyObject *val1, Py_hash_t hash2, PyObject *key2, PyObject *val2, PyObject *transient_id) {
    Py_hash_t hash1 = PyObject_Hash(key1);
    if (hash1 == -1 && PyErr_Occurred()) return NULL;

    if (hash1 == hash2) {
        PyObject *arr = PyList_New(4);
        if (!arr) return NULL;
        Py_INCREF(key1); PyList_SET_ITEM(arr, 0, key1);
        Py_INCREF(val1); PyList_SET_ITEM(arr, 1, val1);
        Py_INCREF(key2); PyList_SET_ITEM(arr, 2, key2);
        Py_INCREF(val2); PyList_SET_ITEM(arr, 3, val2);
        HashCollisionNode *node = HashCollisionNode_create(hash1, 2, arr, transient_id);
        Py_DECREF(arr);
        return (PyObject *)node;
    }

    PyObject *added_leaf = PyList_New(0);
    if (!added_leaf) return NULL;

    PyObject *n1 = BitmapIndexedNode_assoc(EMPTY_BIN, shift, hash1, key1, val1, added_leaf, transient_id);
    if (!n1) {
        Py_DECREF(added_leaf);
        return NULL;
    }

    PyObject *n2;
    if (PyObject_TypeCheck(n1, &BitmapIndexedNodeType)) {
        n2 = BitmapIndexedNode_assoc((BitmapIndexedNode *)n1, shift, hash2, key2, val2, added_leaf, transient_id);
    } else if (PyObject_TypeCheck(n1, &ArrayNodeType)) {
        n2 = ArrayNode_assoc((ArrayNode *)n1, shift, hash2, key2, val2, added_leaf, transient_id);
    } else {
        n2 = HashCollisionNode_assoc((HashCollisionNode *)n1, shift, hash2, key2, val2, added_leaf, transient_id);
    }

    Py_DECREF(added_leaf);
    Py_DECREF(n1);
    return n2;
}

// BitmapIndexedNode implementation
PyObject *BitmapIndexedNode_assoc(BitmapIndexedNode *self, int shift, Py_hash_t hash_val, PyObject *key, PyObject *val, PyObject *added_leaf, PyObject *transient_id) {
    unsigned int bit = bitpos(hash_val, shift);
    int idx = bitmap_index(self->bitmap, bit);

    if (self->bitmap & bit) {
        // Slot exists
        PyObject *key_or_null = PyList_GET_ITEM(self->array, 2 * idx);
        PyObject *val_or_node = PyList_GET_ITEM(self->array, 2 * idx + 1);

        if (key_or_null == Py_None) {
            // Child node
            PyObject *n;
            if (PyObject_TypeCheck(val_or_node, &BitmapIndexedNodeType)) {
                n = BitmapIndexedNode_assoc((BitmapIndexedNode *)val_or_node, shift + BITS, hash_val, key, val, added_leaf, transient_id);
            } else if (PyObject_TypeCheck(val_or_node, &ArrayNodeType)) {
                n = ArrayNode_assoc((ArrayNode *)val_or_node, shift + BITS, hash_val, key, val, added_leaf, transient_id);
            } else {
                n = HashCollisionNode_assoc((HashCollisionNode *)val_or_node, shift + BITS, hash_val, key, val, added_leaf, transient_id);
            }
            if (!n) return NULL;

            if (n == val_or_node) {
                Py_DECREF(n);
                Py_INCREF(self);
                return (PyObject *)self;
            }

            BitmapIndexedNode *node = BitmapIndexedNode_ensure_editable(self, transient_id);
            if (!node) {
                Py_DECREF(n);
                return NULL;
            }
            PyList_SET_ITEM(node->array, 2 * idx + 1, n);
            return (PyObject *)node;
        }

        int eq = PyObject_RichCompareBool(key, key_or_null, Py_EQ);
        if (eq < 0) return NULL;

        if (eq) {
            // Same key - update value
            if (val == val_or_node) {
                Py_INCREF(self);
                return (PyObject *)self;
            }
            BitmapIndexedNode *node = BitmapIndexedNode_ensure_editable(self, transient_id);
            if (!node) return NULL;
            Py_INCREF(val);
            Py_DECREF(PyList_GET_ITEM(node->array, 2 * idx + 1));
            PyList_SET_ITEM(node->array, 2 * idx + 1, val);
            return (PyObject *)node;
        }

        // Hash collision at this level - need to go deeper
        if (PyList_Append(added_leaf, Py_True) < 0) return NULL;

        PyObject *new_node = create_node(shift + BITS, key_or_null, val_or_node, hash_val, key, val, transient_id);
        if (!new_node) return NULL;

        BitmapIndexedNode *node = BitmapIndexedNode_ensure_editable(self, transient_id);
        if (!node) {
            Py_DECREF(new_node);
            return NULL;
        }
        Py_INCREF(Py_None);
        Py_DECREF(PyList_GET_ITEM(node->array, 2 * idx));
        PyList_SET_ITEM(node->array, 2 * idx, Py_None);
        Py_DECREF(PyList_GET_ITEM(node->array, 2 * idx + 1));
        PyList_SET_ITEM(node->array, 2 * idx + 1, new_node);
        return (PyObject *)node;
    } else {
        // New slot
        int n = ctpop(self->bitmap);
        if (n >= WIDTH / 2) {
            // Upgrade to ArrayNode
            PyObject *nodes = PyList_New(WIDTH);
            if (!nodes) return NULL;
            for (int i = 0; i < WIDTH; i++) {
                Py_INCREF(Py_None);
                PyList_SET_ITEM(nodes, i, Py_None);
            }

            int jdx = mask_hash(hash_val, shift);
            PyObject *new_bin = BitmapIndexedNode_assoc(EMPTY_BIN, shift + BITS, hash_val, key, val, added_leaf, transient_id);
            if (!new_bin) {
                Py_DECREF(nodes);
                return NULL;
            }
            Py_DECREF(PyList_GET_ITEM(nodes, jdx));
            PyList_SET_ITEM(nodes, jdx, new_bin);

            int j = 0;
            for (int i = 0; i < WIDTH; i++) {
                if ((self->bitmap >> i) & 1) {
                    PyObject *k = PyList_GET_ITEM(self->array, j);
                    PyObject *v = PyList_GET_ITEM(self->array, j + 1);
                    if (k == Py_None) {
                        Py_INCREF(v);
                        Py_DECREF(PyList_GET_ITEM(nodes, i));
                        PyList_SET_ITEM(nodes, i, v);
                    } else {
                        PyObject *al = PyList_New(0);
                        Py_hash_t kh = PyObject_Hash(k);
                        if (kh == -1 && PyErr_Occurred()) {
                            Py_DECREF(al);
                            Py_DECREF(nodes);
                            return NULL;
                        }
                        PyObject *child = BitmapIndexedNode_assoc(EMPTY_BIN, shift + BITS, kh, k, v, al, transient_id);
                        Py_DECREF(al);
                        if (!child) {
                            Py_DECREF(nodes);
                            return NULL;
                        }
                        Py_DECREF(PyList_GET_ITEM(nodes, i));
                        PyList_SET_ITEM(nodes, i, child);
                    }
                    j += 2;
                }
            }

            ArrayNode *result = ArrayNode_create(n + 1, nodes, transient_id);
            Py_DECREF(nodes);
            return (PyObject *)result;
        } else {
            // Insert into bitmap node
            if (PyList_Append(added_leaf, Py_True) < 0) return NULL;

            Py_ssize_t arr_len = PyList_Size(self->array);
            PyObject *new_array = PyList_New(arr_len + 2);
            if (!new_array) return NULL;

            for (Py_ssize_t i = 0; i < 2 * idx; i++) {
                PyObject *item = PyList_GET_ITEM(self->array, i);
                Py_INCREF(item);
                PyList_SET_ITEM(new_array, i, item);
            }
            Py_INCREF(key);
            PyList_SET_ITEM(new_array, 2 * idx, key);
            Py_INCREF(val);
            PyList_SET_ITEM(new_array, 2 * idx + 1, val);
            for (Py_ssize_t i = 2 * idx; i < arr_len; i++) {
                PyObject *item = PyList_GET_ITEM(self->array, i);
                Py_INCREF(item);
                PyList_SET_ITEM(new_array, i + 2, item);
            }

            BitmapIndexedNode *node = BitmapIndexedNode_create(self->bitmap | bit, new_array, transient_id);
            Py_DECREF(new_array);
            return (PyObject *)node;
        }
    }
}

PyObject *BitmapIndexedNode_find(BitmapIndexedNode *self, int shift, Py_hash_t hash_val, PyObject *key, PyObject *not_found) {
    unsigned int bit = bitpos(hash_val, shift);
    if ((self->bitmap & bit) == 0) {
        Py_INCREF(not_found);
        return not_found;
    }

    int idx = bitmap_index(self->bitmap, bit);
    PyObject *key_or_null = PyList_GET_ITEM(self->array, 2 * idx);
    PyObject *val_or_node = PyList_GET_ITEM(self->array, 2 * idx + 1);

    if (key_or_null == Py_None) {
        if (PyObject_TypeCheck(val_or_node, &BitmapIndexedNodeType)) {
            return BitmapIndexedNode_find((BitmapIndexedNode *)val_or_node, shift + BITS, hash_val, key, not_found);
        } else if (PyObject_TypeCheck(val_or_node, &ArrayNodeType)) {
            return ArrayNode_find((ArrayNode *)val_or_node, shift + BITS, hash_val, key, not_found);
        } else {
            return HashCollisionNode_find((HashCollisionNode *)val_or_node, shift + BITS, hash_val, key, not_found);
        }
    }

    int eq = PyObject_RichCompareBool(key, key_or_null, Py_EQ);
    if (eq < 0) return NULL;

    if (eq) {
        Py_INCREF(val_or_node);
        return val_or_node;
    }

    Py_INCREF(not_found);
    return not_found;
}

PyObject *BitmapIndexedNode_dissoc(BitmapIndexedNode *self, int shift, Py_hash_t hash_val, PyObject *key, PyObject *removed_leaf, PyObject *transient_id) {
    unsigned int bit = bitpos(hash_val, shift);
    if ((self->bitmap & bit) == 0) {
        Py_INCREF(self);
        return (PyObject *)self;
    }

    int idx = bitmap_index(self->bitmap, bit);
    PyObject *key_or_null = PyList_GET_ITEM(self->array, 2 * idx);
    PyObject *val_or_node = PyList_GET_ITEM(self->array, 2 * idx + 1);

    if (key_or_null == Py_None) {
        // Recurse into child node
        PyObject *n;
        if (PyObject_TypeCheck(val_or_node, &BitmapIndexedNodeType)) {
            n = BitmapIndexedNode_dissoc((BitmapIndexedNode *)val_or_node, shift + BITS, hash_val, key, removed_leaf, transient_id);
        } else if (PyObject_TypeCheck(val_or_node, &ArrayNodeType)) {
            n = ArrayNode_dissoc((ArrayNode *)val_or_node, shift + BITS, hash_val, key, removed_leaf, transient_id);
        } else {
            n = HashCollisionNode_dissoc((HashCollisionNode *)val_or_node, shift + BITS, hash_val, key, removed_leaf, transient_id);
        }
        if (!n) return NULL;

        if (n == val_or_node) {
            Py_DECREF(n);
            Py_INCREF(self);
            return (PyObject *)self;
        }
        if (n != Py_None) {
            BitmapIndexedNode *node = BitmapIndexedNode_ensure_editable(self, transient_id);
            if (!node) {
                Py_DECREF(n);
                return NULL;
            }
            Py_DECREF(PyList_GET_ITEM(node->array, 2 * idx + 1));
            PyList_SET_ITEM(node->array, 2 * idx + 1, n);
            return (PyObject *)node;
        }
        if (self->bitmap == bit) {
            Py_DECREF(n);
            Py_INCREF(Py_None);
            return Py_None;
        }
        Py_DECREF(n);

        // Remove entry
        Py_ssize_t arr_len = PyList_Size(self->array);
        PyObject *new_array = PyList_New(arr_len - 2);
        if (!new_array) return NULL;

        for (Py_ssize_t i = 0; i < 2 * idx; i++) {
            PyObject *item = PyList_GET_ITEM(self->array, i);
            Py_INCREF(item);
            PyList_SET_ITEM(new_array, i, item);
        }
        for (Py_ssize_t i = 2 * idx + 2; i < arr_len; i++) {
            PyObject *item = PyList_GET_ITEM(self->array, i);
            Py_INCREF(item);
            PyList_SET_ITEM(new_array, i - 2, item);
        }

        BitmapIndexedNode *node = BitmapIndexedNode_create(self->bitmap ^ bit, new_array, transient_id);
        Py_DECREF(new_array);
        return (PyObject *)node;
    }

    int eq = PyObject_RichCompareBool(key, key_or_null, Py_EQ);
    if (eq < 0) return NULL;

    if (eq) {
        // Mark that we found and removed a leaf
        if (removed_leaf && PyList_Append(removed_leaf, Py_True) < 0) return NULL;

        if (self->bitmap == bit) {
            Py_INCREF(Py_None);
            return Py_None;
        }

        Py_ssize_t arr_len = PyList_Size(self->array);
        PyObject *new_array = PyList_New(arr_len - 2);
        if (!new_array) return NULL;

        for (Py_ssize_t i = 0; i < 2 * idx; i++) {
            PyObject *item = PyList_GET_ITEM(self->array, i);
            Py_INCREF(item);
            PyList_SET_ITEM(new_array, i, item);
        }
        for (Py_ssize_t i = 2 * idx + 2; i < arr_len; i++) {
            PyObject *item = PyList_GET_ITEM(self->array, i);
            Py_INCREF(item);
            PyList_SET_ITEM(new_array, i - 2, item);
        }

        BitmapIndexedNode *node = BitmapIndexedNode_create(self->bitmap ^ bit, new_array, transient_id);
        Py_DECREF(new_array);
        return (PyObject *)node;
    }

    Py_INCREF(self);
    return (PyObject *)self;
}

// Iterator mode constants
#define ITER_MODE_ITEMS 0
#define ITER_MODE_KEYS 1
#define ITER_MODE_VALUES 2

static PyObject *hamt_node_iter_mode_guarded(
    PyObject *node, int mode, int owner_guard, uint64_t owner_thread_id) {
    if (PyObject_TypeCheck(node, &BitmapIndexedNodeType)) {
        return owner_guard
            ? BitmapIndexedNode_iter_mode_transient(
                  (BitmapIndexedNode *)node, mode, owner_thread_id)
            : BitmapIndexedNode_iter_mode((BitmapIndexedNode *)node, mode);
    }
    if (PyObject_TypeCheck(node, &ArrayNodeType)) {
        return owner_guard
            ? ArrayNode_iter_mode_transient(
                  (ArrayNode *)node, mode, owner_thread_id)
            : ArrayNode_iter_mode((ArrayNode *)node, mode);
    }
    return owner_guard
        ? HashCollisionNode_iter_mode_transient(
              (HashCollisionNode *)node, mode, owner_thread_id)
        : HashCollisionNode_iter_mode((HashCollisionNode *)node, mode);
}

// BitmapIndexedNode iterator
typedef struct {
    PyObject_HEAD
    BitmapIndexedNode *node;
    Py_ssize_t index;
    PyObject *child_iter;
    int mode;  // ITER_MODE_ITEMS, ITER_MODE_KEYS, or ITER_MODE_VALUES
    int busy;
    int owner_guard;
    uint64_t owner_thread_id;
} BitmapIndexedNodeIterator;

PyTypeObject BitmapIndexedNodeIteratorType;

static int BitmapIndexedNodeIterator_traverse(
    BitmapIndexedNodeIterator *self, visitproc visit, void *arg) {
    Py_VISIT(self->node);
    Py_VISIT(self->child_iter);
    return 0;
}

static int BitmapIndexedNodeIterator_clear(BitmapIndexedNodeIterator *self) {
    Py_CLEAR(self->node);
    Py_CLEAR(self->child_iter);
    return 0;
}

static void BitmapIndexedNodeIterator_dealloc(BitmapIndexedNodeIterator *self) {
    PyObject_GC_UnTrack(self);
    BitmapIndexedNodeIterator_clear(self);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *BitmapIndexedNodeIterator_next_impl(
    BitmapIndexedNodeIterator *self) {
    if (self->node == NULL || self->node->array == NULL) return NULL;

    // If we have a child iterator, try to get next from it
    while (self->child_iter) {
        PyObject *result = PyIter_Next(self->child_iter);
        if (result) return result;
        if (PyErr_Occurred()) return NULL;
        Py_CLEAR(self->child_iter);
    }

    Py_ssize_t arr_len = PyList_Size(self->node->array);
    while (self->index < arr_len) {
        PyObject *key_or_null = PyList_GET_ITEM(self->node->array, self->index);
        PyObject *val_or_node = PyList_GET_ITEM(self->node->array, self->index + 1);
        self->index += 2;

        if (key_or_null != Py_None) {
            // Direct key-value pair - return based on mode
            PyObject *result;
            switch (self->mode) {
                case ITER_MODE_KEYS:
                    result = key_or_null;
                    Py_INCREF(result);
                    break;
                case ITER_MODE_VALUES:
                    result = val_or_node;
                    Py_INCREF(result);
                    break;
                default:  // ITER_MODE_ITEMS
                    result = PyTuple_Pack(2, key_or_null, val_or_node);
                    break;
            }
            return result;
        } else if (val_or_node != Py_None) {
            // Child iterators inherit transient ownership, when present.
            self->child_iter = hamt_node_iter_mode_guarded(
                val_or_node, self->mode, self->owner_guard,
                self->owner_thread_id);
            if (!self->child_iter) return NULL;

            PyObject *result = PyIter_Next(self->child_iter);
            if (result) return result;
            if (PyErr_Occurred()) return NULL;
            Py_CLEAR(self->child_iter);
        }
    }

    return NULL;  // StopIteration
}

static PyObject *BitmapIndexedNodeIterator_next(
    BitmapIndexedNodeIterator *self) {
    PyObject *result = NULL;

    if (self->owner_guard &&
        pds_check_transient_owner(self->owner_thread_id) < 0) {
        return NULL;
    }

    PDS_BEGIN_CRITICAL_SECTION(self);
    if (self->busy) {
        PyErr_SetString(PyExc_RuntimeError, PDS_ITERATOR_BUSY_ERROR);
    } else {
        self->busy = 1;
        result = BitmapIndexedNodeIterator_next_impl(self);
        self->busy = 0;
    }
    PDS_END_CRITICAL_SECTION();

    return result;
}

PyTypeObject BitmapIndexedNodeIteratorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.BitmapIndexedNodeIterator",
    .tp_basicsize = sizeof(BitmapIndexedNodeIterator),
    .tp_dealloc = (destructor)BitmapIndexedNodeIterator_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .tp_traverse = (traverseproc)BitmapIndexedNodeIterator_traverse,
    .tp_clear = (inquiry)BitmapIndexedNodeIterator_clear,
    .tp_iter = PyObject_SelfIter,
    .tp_alloc = PyType_GenericAlloc,
    .tp_free = PyObject_GC_Del,
    .tp_iternext = (iternextfunc)BitmapIndexedNodeIterator_next,
};

static PyObject *BitmapIndexedNode_iter_mode_impl(
    BitmapIndexedNode *self, int mode, int owner_guard,
    uint64_t owner_thread_id) {
    BitmapIndexedNodeIterator *it =
        (BitmapIndexedNodeIterator *)BitmapIndexedNodeIteratorType.tp_alloc(
            &BitmapIndexedNodeIteratorType, 0);
    if (!it) return NULL;

    it->node = self;
    Py_INCREF(self);
    it->index = 0;
    it->child_iter = NULL;
    it->mode = mode;
    it->busy = 0;
    it->owner_guard = owner_guard;
    it->owner_thread_id = owner_thread_id;
    return (PyObject *)it;
}

PyObject *BitmapIndexedNode_iter_mode(BitmapIndexedNode *self, int mode) {
    return BitmapIndexedNode_iter_mode_impl(self, mode, 0, 0);
}

PyObject *BitmapIndexedNode_iter_mode_transient(
    BitmapIndexedNode *self, int mode, uint64_t owner_thread_id) {
    return BitmapIndexedNode_iter_mode_impl(
        self, mode, 1, owner_thread_id);
}

PyObject *BitmapIndexedNode_iter_kv(BitmapIndexedNode *self) {
    return BitmapIndexedNode_iter_mode(self, ITER_MODE_ITEMS);
}

// ArrayNode implementation
PyObject *ArrayNode_assoc(ArrayNode *self, int shift, Py_hash_t hash_val, PyObject *key, PyObject *val, PyObject *added_leaf, PyObject *transient_id) {
    int idx = mask_hash(hash_val, shift);
    PyObject *node = PyList_GET_ITEM(self->array, idx);

    if (node == Py_None) {
        // Use a fresh added_leaf for recursive call to avoid double-counting
        // We'll count this insertion here, not in the recursive call
        PyObject *al_fresh = PyList_New(0);
        if (!al_fresh) return NULL;

        PyObject *new_node = BitmapIndexedNode_assoc(EMPTY_BIN, shift + BITS, hash_val, key, val, al_fresh, transient_id);
        Py_DECREF(al_fresh);
        if (!new_node) return NULL;

        // Count this as a new insertion
        if (PyList_Append(added_leaf, Py_True) < 0) {
            Py_DECREF(new_node);
            return NULL;
        }

        ArrayNode *editable = ArrayNode_ensure_editable(self, transient_id);
        if (!editable) {
            Py_DECREF(new_node);
            return NULL;
        }
        Py_DECREF(PyList_GET_ITEM(editable->array, idx));
        PyList_SET_ITEM(editable->array, idx, new_node);
        editable->count++;
        return (PyObject *)editable;
    }

    PyObject *n;
    if (PyObject_TypeCheck(node, &BitmapIndexedNodeType)) {
        n = BitmapIndexedNode_assoc((BitmapIndexedNode *)node, shift + BITS, hash_val, key, val, added_leaf, transient_id);
    } else if (PyObject_TypeCheck(node, &ArrayNodeType)) {
        n = ArrayNode_assoc((ArrayNode *)node, shift + BITS, hash_val, key, val, added_leaf, transient_id);
    } else {
        n = HashCollisionNode_assoc((HashCollisionNode *)node, shift + BITS, hash_val, key, val, added_leaf, transient_id);
    }
    if (!n) return NULL;

    if (n == node) {
        Py_DECREF(n);
        Py_INCREF(self);
        return (PyObject *)self;
    }

    ArrayNode *editable = ArrayNode_ensure_editable(self, transient_id);
    if (!editable) {
        Py_DECREF(n);
        return NULL;
    }
    Py_DECREF(PyList_GET_ITEM(editable->array, idx));
    PyList_SET_ITEM(editable->array, idx, n);
    return (PyObject *)editable;
}

PyObject *ArrayNode_find(ArrayNode *self, int shift, Py_hash_t hash_val, PyObject *key, PyObject *not_found) {
    int idx = mask_hash(hash_val, shift);
    PyObject *node = PyList_GET_ITEM(self->array, idx);

    if (node == Py_None) {
        Py_INCREF(not_found);
        return not_found;
    }

    if (PyObject_TypeCheck(node, &BitmapIndexedNodeType)) {
        return BitmapIndexedNode_find((BitmapIndexedNode *)node, shift + BITS, hash_val, key, not_found);
    } else if (PyObject_TypeCheck(node, &ArrayNodeType)) {
        return ArrayNode_find((ArrayNode *)node, shift + BITS, hash_val, key, not_found);
    } else {
        return HashCollisionNode_find((HashCollisionNode *)node, shift + BITS, hash_val, key, not_found);
    }
}

static PyObject *ArrayNode_pack(ArrayNode *self, PyObject *transient_id, int idx) {
    PyObject *new_array = PyList_New(0);
    if (!new_array) return NULL;

    unsigned int bitmap = 0;
    for (int i = 0; i < WIDTH; i++) {
        PyObject *node = PyList_GET_ITEM(self->array, i);
        if (i != idx && node != Py_None) {
            if (PyList_Append(new_array, Py_None) < 0) {
                Py_DECREF(new_array);
                return NULL;
            }
            if (PyList_Append(new_array, node) < 0) {
                Py_DECREF(new_array);
                return NULL;
            }
            bitmap |= 1U << i;
        }
    }

    BitmapIndexedNode *result = BitmapIndexedNode_create(bitmap, new_array, transient_id);
    Py_DECREF(new_array);
    return (PyObject *)result;
}

PyObject *ArrayNode_dissoc(ArrayNode *self, int shift, Py_hash_t hash_val, PyObject *key, PyObject *removed_leaf, PyObject *transient_id) {
    int idx = mask_hash(hash_val, shift);
    PyObject *node = PyList_GET_ITEM(self->array, idx);

    if (node == Py_None) {
        Py_INCREF(self);
        return (PyObject *)self;
    }

    PyObject *n;
    if (PyObject_TypeCheck(node, &BitmapIndexedNodeType)) {
        n = BitmapIndexedNode_dissoc((BitmapIndexedNode *)node, shift + BITS, hash_val, key, removed_leaf, transient_id);
    } else if (PyObject_TypeCheck(node, &ArrayNodeType)) {
        n = ArrayNode_dissoc((ArrayNode *)node, shift + BITS, hash_val, key, removed_leaf, transient_id);
    } else {
        n = HashCollisionNode_dissoc((HashCollisionNode *)node, shift + BITS, hash_val, key, removed_leaf, transient_id);
    }
    if (!n) return NULL;

    if (n == node) {
        Py_DECREF(n);
        Py_INCREF(self);
        return (PyObject *)self;
    }

    if (n == Py_None) {
        Py_DECREF(n);
        if (self->count <= WIDTH / 4) {
            return ArrayNode_pack(self, transient_id, idx);
        }
        ArrayNode *editable = ArrayNode_ensure_editable(self, transient_id);
        if (!editable) return NULL;
        Py_INCREF(Py_None);
        Py_DECREF(PyList_GET_ITEM(editable->array, idx));
        PyList_SET_ITEM(editable->array, idx, Py_None);
        editable->count--;
        return (PyObject *)editable;
    }

    ArrayNode *editable = ArrayNode_ensure_editable(self, transient_id);
    if (!editable) {
        Py_DECREF(n);
        return NULL;
    }
    Py_DECREF(PyList_GET_ITEM(editable->array, idx));
    PyList_SET_ITEM(editable->array, idx, n);
    return (PyObject *)editable;
}

// ArrayNode iterator
typedef struct {
    PyObject_HEAD
    ArrayNode *node;
    int index;
    PyObject *child_iter;
    int mode;  // ITER_MODE_ITEMS, ITER_MODE_KEYS, or ITER_MODE_VALUES
    int busy;
    int owner_guard;
    uint64_t owner_thread_id;
} ArrayNodeIterator;

PyTypeObject ArrayNodeIteratorType;

static int ArrayNodeIterator_traverse(
    ArrayNodeIterator *self, visitproc visit, void *arg) {
    Py_VISIT(self->node);
    Py_VISIT(self->child_iter);
    return 0;
}

static int ArrayNodeIterator_clear(ArrayNodeIterator *self) {
    Py_CLEAR(self->node);
    Py_CLEAR(self->child_iter);
    return 0;
}

static void ArrayNodeIterator_dealloc(ArrayNodeIterator *self) {
    PyObject_GC_UnTrack(self);
    ArrayNodeIterator_clear(self);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *ArrayNodeIterator_next_impl(ArrayNodeIterator *self) {
    if (self->node == NULL || self->node->array == NULL) return NULL;

    while (self->child_iter) {
        PyObject *result = PyIter_Next(self->child_iter);
        if (result) return result;
        if (PyErr_Occurred()) return NULL;
        Py_CLEAR(self->child_iter);
    }

    while (self->index < WIDTH) {
        PyObject *node = PyList_GET_ITEM(self->node->array, self->index);
        self->index++;

        if (node != Py_None) {
            self->child_iter = hamt_node_iter_mode_guarded(
                node, self->mode, self->owner_guard,
                self->owner_thread_id);
            if (!self->child_iter) return NULL;

            PyObject *result = PyIter_Next(self->child_iter);
            if (result) return result;
            if (PyErr_Occurred()) return NULL;
            Py_CLEAR(self->child_iter);
        }
    }

    return NULL;
}

static PyObject *ArrayNodeIterator_next(ArrayNodeIterator *self) {
    PyObject *result = NULL;

    if (self->owner_guard &&
        pds_check_transient_owner(self->owner_thread_id) < 0) {
        return NULL;
    }

    PDS_BEGIN_CRITICAL_SECTION(self);
    if (self->busy) {
        PyErr_SetString(PyExc_RuntimeError, PDS_ITERATOR_BUSY_ERROR);
    } else {
        self->busy = 1;
        result = ArrayNodeIterator_next_impl(self);
        self->busy = 0;
    }
    PDS_END_CRITICAL_SECTION();

    return result;
}

PyTypeObject ArrayNodeIteratorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.ArrayNodeIterator",
    .tp_basicsize = sizeof(ArrayNodeIterator),
    .tp_dealloc = (destructor)ArrayNodeIterator_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .tp_traverse = (traverseproc)ArrayNodeIterator_traverse,
    .tp_clear = (inquiry)ArrayNodeIterator_clear,
    .tp_iter = PyObject_SelfIter,
    .tp_alloc = PyType_GenericAlloc,
    .tp_free = PyObject_GC_Del,
    .tp_iternext = (iternextfunc)ArrayNodeIterator_next,
};

static PyObject *ArrayNode_iter_mode_impl(
    ArrayNode *self, int mode, int owner_guard, uint64_t owner_thread_id) {
    ArrayNodeIterator *it = (ArrayNodeIterator *)ArrayNodeIteratorType.tp_alloc(
        &ArrayNodeIteratorType, 0);
    if (!it) return NULL;

    it->node = self;
    Py_INCREF(self);
    it->index = 0;
    it->child_iter = NULL;
    it->mode = mode;
    it->busy = 0;
    it->owner_guard = owner_guard;
    it->owner_thread_id = owner_thread_id;
    return (PyObject *)it;
}

PyObject *ArrayNode_iter_mode(ArrayNode *self, int mode) {
    return ArrayNode_iter_mode_impl(self, mode, 0, 0);
}

PyObject *ArrayNode_iter_mode_transient(
    ArrayNode *self, int mode, uint64_t owner_thread_id) {
    return ArrayNode_iter_mode_impl(self, mode, 1, owner_thread_id);
}

PyObject *ArrayNode_iter_kv(ArrayNode *self) {
    return ArrayNode_iter_mode(self, ITER_MODE_ITEMS);
}

// HashCollisionNode implementation
PyObject *HashCollisionNode_assoc(HashCollisionNode *self, int shift, Py_hash_t hash_val, PyObject *key, PyObject *val, PyObject *added_leaf, PyObject *transient_id) {
    if (hash_val == self->hash) {
        int idx = HashCollisionNode_find_index(self, key);
        if (idx == -2) return NULL;  // Error

        if (idx != -1) {
            PyObject *existing = PyList_GET_ITEM(self->array, idx + 1);
            if (existing == val) {
                Py_INCREF(self);
                return (PyObject *)self;
            }
            HashCollisionNode *node = HashCollisionNode_ensure_editable(self, transient_id);
            if (!node) return NULL;
            Py_INCREF(val);
            Py_DECREF(PyList_GET_ITEM(node->array, idx + 1));
            PyList_SET_ITEM(node->array, idx + 1, val);
            return (PyObject *)node;
        }

        if (PyList_Append(added_leaf, Py_True) < 0) return NULL;

        Py_ssize_t arr_len = PyList_Size(self->array);
        PyObject *new_array = PyList_New(arr_len + 2);
        if (!new_array) return NULL;

        for (Py_ssize_t i = 0; i < arr_len; i++) {
            PyObject *item = PyList_GET_ITEM(self->array, i);
            Py_INCREF(item);
            PyList_SET_ITEM(new_array, i, item);
        }
        Py_INCREF(key);
        PyList_SET_ITEM(new_array, arr_len, key);
        Py_INCREF(val);
        PyList_SET_ITEM(new_array, arr_len + 1, val);

        HashCollisionNode *node = HashCollisionNode_create(self->hash, self->count + 1, new_array, transient_id);
        Py_DECREF(new_array);
        return (PyObject *)node;
    }

    // Different hash - nest in a bitmap node
    PyObject *arr = PyList_New(2);
    if (!arr) return NULL;
    Py_INCREF(Py_None);
    PyList_SET_ITEM(arr, 0, Py_None);
    Py_INCREF(self);
    PyList_SET_ITEM(arr, 1, (PyObject *)self);

    BitmapIndexedNode *bin = BitmapIndexedNode_create(bitpos(self->hash, shift), arr, transient_id);
    Py_DECREF(arr);
    if (!bin) return NULL;

    PyObject *result = BitmapIndexedNode_assoc(bin, shift, hash_val, key, val, added_leaf, transient_id);
    Py_DECREF(bin);
    return result;
}

PyObject *HashCollisionNode_find(HashCollisionNode *self, int shift, Py_hash_t hash_val, PyObject *key, PyObject *not_found) {
    int idx = HashCollisionNode_find_index(self, key);
    if (idx == -2) return NULL;
    if (idx < 0) {
        Py_INCREF(not_found);
        return not_found;
    }
    PyObject *result = PyList_GET_ITEM(self->array, idx + 1);
    Py_INCREF(result);
    return result;
}

PyObject *HashCollisionNode_dissoc(HashCollisionNode *self, int shift, Py_hash_t hash_val, PyObject *key, PyObject *removed_leaf, PyObject *transient_id) {
    int idx = HashCollisionNode_find_index(self, key);
    if (idx == -2) return NULL;
    if (idx == -1) {
        Py_INCREF(self);
        return (PyObject *)self;
    }

    // Mark that we found and removed a leaf
    if (removed_leaf && PyList_Append(removed_leaf, Py_True) < 0) return NULL;

    if (self->count == 1) {
        Py_INCREF(Py_None);
        return Py_None;
    }

    Py_ssize_t arr_len = PyList_Size(self->array);
    PyObject *new_array = PyList_New(arr_len - 2);
    if (!new_array) return NULL;

    Py_ssize_t j = 0;
    for (Py_ssize_t i = 0; i < arr_len; i += 2) {
        if (i != idx) {
            PyObject *k = PyList_GET_ITEM(self->array, i);
            PyObject *v = PyList_GET_ITEM(self->array, i + 1);
            Py_INCREF(k);
            Py_INCREF(v);
            PyList_SET_ITEM(new_array, j, k);
            PyList_SET_ITEM(new_array, j + 1, v);
            j += 2;
        }
    }

    HashCollisionNode *node = HashCollisionNode_create(self->hash, self->count - 1, new_array, transient_id);
    Py_DECREF(new_array);
    return (PyObject *)node;
}

// HashCollisionNode iterator
typedef struct {
    PyObject_HEAD
    HashCollisionNode *node;
    Py_ssize_t index;
    int mode;  // ITER_MODE_ITEMS, ITER_MODE_KEYS, or ITER_MODE_VALUES
    int busy;
    int owner_guard;
    uint64_t owner_thread_id;
} HashCollisionNodeIterator;

PyTypeObject HashCollisionNodeIteratorType;

static int HashCollisionNodeIterator_traverse(
    HashCollisionNodeIterator *self, visitproc visit, void *arg) {
    Py_VISIT(self->node);
    return 0;
}

static int HashCollisionNodeIterator_clear(HashCollisionNodeIterator *self) {
    Py_CLEAR(self->node);
    return 0;
}

static void HashCollisionNodeIterator_dealloc(HashCollisionNodeIterator *self) {
    PyObject_GC_UnTrack(self);
    HashCollisionNodeIterator_clear(self);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *HashCollisionNodeIterator_next_impl(
    HashCollisionNodeIterator *self) {
    if (self->node == NULL || self->node->array == NULL ||
        self->index >= PyList_Size(self->node->array)) {
        return NULL;
    }

    PyObject *key = PyList_GET_ITEM(self->node->array, self->index);
    PyObject *val = PyList_GET_ITEM(self->node->array, self->index + 1);
    self->index += 2;

    PyObject *result;
    switch (self->mode) {
        case ITER_MODE_KEYS:
            result = key;
            Py_INCREF(result);
            break;
        case ITER_MODE_VALUES:
            result = val;
            Py_INCREF(result);
            break;
        default:  // ITER_MODE_ITEMS
            result = PyTuple_Pack(2, key, val);
            break;
    }
    return result;
}

static PyObject *HashCollisionNodeIterator_next(
    HashCollisionNodeIterator *self) {
    PyObject *result = NULL;

    if (self->owner_guard &&
        pds_check_transient_owner(self->owner_thread_id) < 0) {
        return NULL;
    }

    PDS_BEGIN_CRITICAL_SECTION(self);
    if (self->busy) {
        PyErr_SetString(PyExc_RuntimeError, PDS_ITERATOR_BUSY_ERROR);
    } else {
        self->busy = 1;
        result = HashCollisionNodeIterator_next_impl(self);
        self->busy = 0;
    }
    PDS_END_CRITICAL_SECTION();

    return result;
}

PyTypeObject HashCollisionNodeIteratorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.HashCollisionNodeIterator",
    .tp_basicsize = sizeof(HashCollisionNodeIterator),
    .tp_dealloc = (destructor)HashCollisionNodeIterator_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .tp_traverse = (traverseproc)HashCollisionNodeIterator_traverse,
    .tp_clear = (inquiry)HashCollisionNodeIterator_clear,
    .tp_iter = PyObject_SelfIter,
    .tp_alloc = PyType_GenericAlloc,
    .tp_free = PyObject_GC_Del,
    .tp_iternext = (iternextfunc)HashCollisionNodeIterator_next,
};

static PyObject *HashCollisionNode_iter_mode_impl(
    HashCollisionNode *self, int mode, int owner_guard,
    uint64_t owner_thread_id) {
    HashCollisionNodeIterator *it =
        (HashCollisionNodeIterator *)HashCollisionNodeIteratorType.tp_alloc(
            &HashCollisionNodeIteratorType, 0);
    if (!it) return NULL;

    it->node = self;
    Py_INCREF(self);
    it->index = 0;
    it->mode = mode;
    it->busy = 0;
    it->owner_guard = owner_guard;
    it->owner_thread_id = owner_thread_id;
    return (PyObject *)it;
}

PyObject *HashCollisionNode_iter_mode(HashCollisionNode *self, int mode) {
    return HashCollisionNode_iter_mode_impl(self, mode, 0, 0);
}

PyObject *HashCollisionNode_iter_mode_transient(
    HashCollisionNode *self, int mode, uint64_t owner_thread_id) {
    return HashCollisionNode_iter_mode_impl(
        self, mode, 1, owner_thread_id);
}

PyObject *HashCollisionNode_iter_kv(HashCollisionNode *self) {
    return HashCollisionNode_iter_mode(self, ITER_MODE_ITEMS);
}
