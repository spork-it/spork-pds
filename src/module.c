#include "pds_internal.h"

/* Per-module references to the process-wide immutable singletons. */
typedef struct {
    PyObject *_MISSING;
    PyObject *EMPTY_VECTOR;
    PyObject *EMPTY_DOUBLE_VECTOR;
    PyObject *EMPTY_LONG_VECTOR;
    PyObject *EMPTY_MAP;
    PyObject *EMPTY_SET;
    PyObject *EMPTY_SORTED_VECTOR;
    PyObject *EMPTY_NODE;
    PyObject *EMPTY_DOUBLE_NODE;
    PyObject *EMPTY_LONG_NODE;
    PyObject *EMPTY_BIN;
} PdsState;

static int _singletons_initialized = 0;

static inline PdsState *
pds_get_state(PyObject *module)
{
    return (PdsState *)PyModule_GetState(module);
}

static PyMethodDef pds_methods[] = {
    {"cons", pds_cons, METH_VARARGS, "Create a cons cell"},
    {"vec", pds_vec, METH_VARARGS, "Create a persistent vector"},
    {"vec_f64", pds_vec_f64, METH_VARARGS, "Create a persistent vector of float64"},
    {"vec_i64", pds_vec_i64, METH_VARARGS, "Create a persistent vector of int64"},
    {"hash_map", pds_hash_map, METH_VARARGS, "Create a persistent map from key-value pairs"},
    {"hash_set", pds_set, METH_VARARGS, "Create a persistent set from an iterable"},
    {"sorted_vec", (PyCFunction)pds_sorted_vec, METH_VARARGS | METH_KEYWORDS, "Create a persistent sorted vector"},
    {NULL, NULL, 0, NULL}
};

// =============================================================================
// MODULE STATE MANAGEMENT (GC integration)
// =============================================================================

static int pds_traverse(PyObject *m, visitproc visit, void *arg) {
    PdsState *st = pds_get_state(m);
    if (st == NULL) return 0;

#if !PDS_SINGLETONS_ARE_IMMORTAL
    Py_VISIT(st->_MISSING);
    Py_VISIT(st->EMPTY_VECTOR);
    Py_VISIT(st->EMPTY_DOUBLE_VECTOR);
    Py_VISIT(st->EMPTY_LONG_VECTOR);
    Py_VISIT(st->EMPTY_MAP);
    Py_VISIT(st->EMPTY_SET);
    Py_VISIT(st->EMPTY_SORTED_VECTOR);
    Py_VISIT(st->EMPTY_NODE);
    Py_VISIT(st->EMPTY_DOUBLE_NODE);
    Py_VISIT(st->EMPTY_LONG_NODE);
    Py_VISIT(st->EMPTY_BIN);
#endif

    return 0;
}

static int pds_clear(PyObject *m) {
    PdsState *st = pds_get_state(m);
    if (st == NULL) return 0;

#if !PDS_SINGLETONS_ARE_IMMORTAL
    Py_CLEAR(st->_MISSING);
    Py_CLEAR(st->EMPTY_VECTOR);
    Py_CLEAR(st->EMPTY_DOUBLE_VECTOR);
    Py_CLEAR(st->EMPTY_LONG_VECTOR);
    Py_CLEAR(st->EMPTY_MAP);
    Py_CLEAR(st->EMPTY_SET);
    Py_CLEAR(st->EMPTY_SORTED_VECTOR);
    Py_CLEAR(st->EMPTY_NODE);
    Py_CLEAR(st->EMPTY_DOUBLE_NODE);
    Py_CLEAR(st->EMPTY_LONG_NODE);
    Py_CLEAR(st->EMPTY_BIN);
#else
    st->_MISSING = NULL;
    st->EMPTY_VECTOR = NULL;
    st->EMPTY_DOUBLE_VECTOR = NULL;
    st->EMPTY_LONG_VECTOR = NULL;
    st->EMPTY_MAP = NULL;
    st->EMPTY_SET = NULL;
    st->EMPTY_SORTED_VECTOR = NULL;
    st->EMPTY_NODE = NULL;
    st->EMPTY_DOUBLE_NODE = NULL;
    st->EMPTY_LONG_NODE = NULL;
    st->EMPTY_BIN = NULL;
#endif

    return 0;
}

static void pds_free(void *m) {
#if !PDS_SINGLETONS_ARE_IMMORTAL
    pds_clear((PyObject *)m);
#endif
}

static int pds_exec(PyObject *m);

static PyModuleDef_Slot pds_slots[] = {
    {Py_mod_exec, pds_exec},
#if PY_VERSION_HEX >= 0x030D0000
    // Python 3.13+: Declare this module as free-threading safe
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, NULL}
};

static struct PyModuleDef pdsmodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "spork_pds",
    .m_doc = "Persistent Data Structures - C implementation",
    .m_size = sizeof(PdsState),
    .m_methods = pds_methods,
    .m_slots = pds_slots,
    .m_traverse = pds_traverse,
    .m_clear = pds_clear,
    .m_free = pds_free,
};

static int pds_exec(PyObject *m) {
    PdsState *st = pds_get_state(m);
    if (st == NULL) {
        return -1;
    }

    // Initialize sentinel type first (used for identity objects)
    if (PyType_Ready(&PdsSentinelType) < 0) return -1;

    // Initialize types
    if (PyType_Ready(&ConsType) < 0) return -1;
    if (PyType_Ready(&ConsIteratorType) < 0) return -1;
    if (PyType_Ready(&VectorNodeType) < 0) return -1;
    if (PyType_Ready(&VectorType) < 0) return -1;
    if (PyType_Ready(&VectorIteratorType) < 0) return -1;
    if (PyType_Ready(&TransientVectorType) < 0) return -1;
    if (PyType_Ready(&TransientVectorIteratorType) < 0) return -1;

    // Initialize SortedVector types
    if (PyType_Ready(&RBNodeType) < 0) return -1;
    if (PyType_Ready(&SortedVectorType) < 0) return -1;
    if (PyType_Ready(&SortedVectorIteratorType) < 0) return -1;
    if (PyType_Ready(&TransientSortedVectorType) < 0) return -1;

    // Initialize type-specialized vector types
    if (PyType_Ready(&DoubleVectorNodeType) < 0) return -1;
    if (PyType_Ready(&DoubleVectorType) < 0) return -1;
    if (PyType_Ready(&DoubleVectorIteratorType) < 0) return -1;
    if (PyType_Ready(&TransientDoubleVectorType) < 0) return -1;
    if (PyType_Ready(&IntVectorNodeType) < 0) return -1;
    if (PyType_Ready(&IntVectorType) < 0) return -1;
    if (PyType_Ready(&IntVectorIteratorType) < 0) return -1;
    if (PyType_Ready(&TransientIntVectorType) < 0) return -1;

    if (PyType_Ready(&BitmapIndexedNodeType) < 0) return -1;
    if (PyType_Ready(&ArrayNodeType) < 0) return -1;
    if (PyType_Ready(&HashCollisionNodeType) < 0) return -1;
    if (PyType_Ready(&BitmapIndexedNodeIteratorType) < 0) return -1;
    if (PyType_Ready(&ArrayNodeIteratorType) < 0) return -1;
    if (PyType_Ready(&HashCollisionNodeIteratorType) < 0) return -1;
    if (PyType_Ready(&MapType) < 0) return -1;
    if (PyType_Ready(&TransientMapType) < 0) return -1;
    if (PyType_Ready(&SetType) < 0) return -1;
    if (PyType_Ready(&TransientSetType) < 0) return -1;
    if (PyType_Ready(&SetIteratorType) < 0) return -1;

    // Create singletons only once to ensure sub-interpreter safety.
    // If already initialized, reuse the existing global singletons.
    if (_singletons_initialized) {
        // Reuse existing singletons - just copy globals to module state
        st->_MISSING = _MISSING;
        st->EMPTY_NODE = (PyObject *)EMPTY_NODE;
        st->EMPTY_VECTOR = (PyObject *)EMPTY_VECTOR;
        st->EMPTY_DOUBLE_NODE = (PyObject *)EMPTY_DOUBLE_NODE;
        st->EMPTY_DOUBLE_VECTOR = (PyObject *)EMPTY_DOUBLE_VECTOR;
        st->EMPTY_LONG_NODE = (PyObject *)EMPTY_LONG_NODE;
        st->EMPTY_LONG_VECTOR = (PyObject *)EMPTY_LONG_VECTOR;
        st->EMPTY_BIN = (PyObject *)EMPTY_BIN;
        st->EMPTY_MAP = (PyObject *)EMPTY_MAP;
        st->EMPTY_SET = (PyObject *)EMPTY_SET;
        st->EMPTY_SORTED_VECTOR = (PyObject *)EMPTY_SORTED_VECTOR;
    } else {
        // First initialization - create all singletons

        st->_MISSING = PyObject_New(PyObject, &PdsSentinelType);
        if (!st->_MISSING) return -1;

        // Create empty node
        st->EMPTY_NODE = (PyObject *)VectorNode_create(NULL);
        if (!st->EMPTY_NODE) return -1;

        // Create empty vector
        st->EMPTY_VECTOR = (PyObject *)Vector_create(0, BITS, (VectorNode *)st->EMPTY_NODE, NULL, NULL);
        if (!st->EMPTY_VECTOR) return -1;

        // Create empty double vector node and vector
        st->EMPTY_DOUBLE_NODE = (PyObject *)DoubleVectorNode_create(NULL);
        if (!st->EMPTY_DOUBLE_NODE) return -1;

        st->EMPTY_DOUBLE_VECTOR = (PyObject *)DoubleVector_create(0, BITS, (DoubleVectorNode *)st->EMPTY_DOUBLE_NODE, NULL, 0, NULL);
        if (!st->EMPTY_DOUBLE_VECTOR) return -1;

        // Create empty long vector node and vector
        st->EMPTY_LONG_NODE = (PyObject *)IntVectorNode_create(NULL);
        if (!st->EMPTY_LONG_NODE) return -1;

        st->EMPTY_LONG_VECTOR = (PyObject *)IntVector_create(0, BITS, (IntVectorNode *)st->EMPTY_LONG_NODE, NULL, 0, NULL);
        if (!st->EMPTY_LONG_VECTOR) return -1;

        // Create empty bitmap indexed node
        st->EMPTY_BIN = (PyObject *)BitmapIndexedNode_create(0, NULL, NULL);
        if (!st->EMPTY_BIN) return -1;

        // Create empty map
        st->EMPTY_MAP = (PyObject *)Map_create(0, NULL, NULL);
        if (!st->EMPTY_MAP) return -1;

        // Create empty set
        st->EMPTY_SET = (PyObject *)Set_create(0, NULL, NULL);
        if (!st->EMPTY_SET) return -1;

        // Create empty sorted vector
        st->EMPTY_SORTED_VECTOR = (PyObject *)PyObject_New(SortedVector, &SortedVectorType);
        if (!st->EMPTY_SORTED_VECTOR) return -1;
        {
            SortedVector *sv = (SortedVector *)st->EMPTY_SORTED_VECTOR;
            sv->root = NULL;
            sv->cnt = 0;
            sv->key_fn = NULL;
            sv->reverse = 0;
        }

        // Immortalize singletons for Python 3.12+ to prevent refcount contention
        // in multi-threaded code. Immortal objects don't have their refcounts modified.
        PDS_SET_IMMORTAL(st->_MISSING);
        PDS_SET_IMMORTAL(st->EMPTY_NODE);
        PDS_SET_IMMORTAL(st->EMPTY_VECTOR);
        PDS_SET_IMMORTAL(st->EMPTY_DOUBLE_NODE);
        PDS_SET_IMMORTAL(st->EMPTY_DOUBLE_VECTOR);
        PDS_SET_IMMORTAL(st->EMPTY_LONG_NODE);
        PDS_SET_IMMORTAL(st->EMPTY_LONG_VECTOR);
        PDS_SET_IMMORTAL(st->EMPTY_BIN);
        PDS_SET_IMMORTAL(st->EMPTY_MAP);
        PDS_SET_IMMORTAL(st->EMPTY_SET);
        PDS_SET_IMMORTAL(st->EMPTY_SORTED_VECTOR);

        // Update global aliases for backward compatibility with existing code
        // These are set once and never change, ensuring sub-interpreter safety
        _MISSING = st->_MISSING;
        EMPTY_NODE = (VectorNode *)st->EMPTY_NODE;
        EMPTY_VECTOR = (Vector *)st->EMPTY_VECTOR;
        EMPTY_DOUBLE_NODE = (DoubleVectorNode *)st->EMPTY_DOUBLE_NODE;
        EMPTY_DOUBLE_VECTOR = (DoubleVector *)st->EMPTY_DOUBLE_VECTOR;
        EMPTY_LONG_NODE = (IntVectorNode *)st->EMPTY_LONG_NODE;
        EMPTY_LONG_VECTOR = (IntVector *)st->EMPTY_LONG_VECTOR;
        EMPTY_BIN = (BitmapIndexedNode *)st->EMPTY_BIN;
        EMPTY_MAP = (Map *)st->EMPTY_MAP;
        EMPTY_SET = (Set *)st->EMPTY_SET;
        EMPTY_SORTED_VECTOR = (SortedVector *)st->EMPTY_SORTED_VECTOR;

        // Mark as initialized
        _singletons_initialized = 1;
    }

    // Add types to module
    Py_INCREF(&ConsType);
    if (PyModule_AddObject(m, "Cons", (PyObject *)&ConsType) < 0) {
        Py_DECREF(&ConsType);
        return -1;
    }

    Py_INCREF(&VectorType);
    if (PyModule_AddObject(m, "Vector", (PyObject *)&VectorType) < 0) {
        Py_DECREF(&VectorType);
        return -1;
    }

    Py_INCREF(&TransientVectorType);
    if (PyModule_AddObject(m, "TransientVector", (PyObject *)&TransientVectorType) < 0) {
        Py_DECREF(&TransientVectorType);
        return -1;
    }

    Py_INCREF(&MapType);
    if (PyModule_AddObject(m, "Map", (PyObject *)&MapType) < 0) {
        Py_DECREF(&MapType);
        return -1;
    }

    Py_INCREF(&TransientMapType);
    if (PyModule_AddObject(m, "TransientMap", (PyObject *)&TransientMapType) < 0) {
        Py_DECREF(&TransientMapType);
        return -1;
    }

    Py_INCREF(&SetType);
    if (PyModule_AddObject(m, "Set", (PyObject *)&SetType) < 0) {
        Py_DECREF(&SetType);
        return -1;
    }

    Py_INCREF(&TransientSetType);
    if (PyModule_AddObject(m, "TransientSet", (PyObject *)&TransientSetType) < 0) {
        Py_DECREF(&TransientSetType);
        return -1;
    }

    // Add empty instances
    Py_INCREF(st->EMPTY_VECTOR);
    if (PyModule_AddObject(m, "EMPTY_VECTOR", st->EMPTY_VECTOR) < 0) {
        Py_DECREF(st->EMPTY_VECTOR);
        return -1;
    }

    Py_INCREF(st->EMPTY_MAP);
    if (PyModule_AddObject(m, "EMPTY_MAP", st->EMPTY_MAP) < 0) {
        Py_DECREF(st->EMPTY_MAP);
        return -1;
    }

    Py_INCREF(st->EMPTY_SET);
    if (PyModule_AddObject(m, "EMPTY_SET", st->EMPTY_SET) < 0) {
        Py_DECREF(st->EMPTY_SET);
        return -1;
    }

    // Add SortedVector types
    Py_INCREF(&SortedVectorType);
    if (PyModule_AddObject(m, "SortedVector", (PyObject *)&SortedVectorType) < 0) {
        Py_DECREF(&SortedVectorType);
        return -1;
    }

    Py_INCREF(&TransientSortedVectorType);
    if (PyModule_AddObject(m, "TransientSortedVector", (PyObject *)&TransientSortedVectorType) < 0) {
        Py_DECREF(&TransientSortedVectorType);
        return -1;
    }

    Py_INCREF(st->EMPTY_SORTED_VECTOR);
    if (PyModule_AddObject(m, "EMPTY_SORTED_VECTOR", st->EMPTY_SORTED_VECTOR) < 0) {
        Py_DECREF(st->EMPTY_SORTED_VECTOR);
        return -1;
    }

    // Add type-specialized vector types
    Py_INCREF(&DoubleVectorType);
    if (PyModule_AddObject(m, "DoubleVector", (PyObject *)&DoubleVectorType) < 0) {
        Py_DECREF(&DoubleVectorType);
        return -1;
    }

    Py_INCREF(&IntVectorType);
    if (PyModule_AddObject(m, "IntVector", (PyObject *)&IntVectorType) < 0) {
        Py_DECREF(&IntVectorType);
        return -1;
    }

    Py_INCREF(&TransientDoubleVectorType);
    if (PyModule_AddObject(m, "TransientDoubleVector", (PyObject *)&TransientDoubleVectorType) < 0) {
        Py_DECREF(&TransientDoubleVectorType);
        return -1;
    }

    Py_INCREF(&TransientIntVectorType);
    if (PyModule_AddObject(m, "TransientIntVector", (PyObject *)&TransientIntVectorType) < 0) {
        Py_DECREF(&TransientIntVectorType);
        return -1;
    }

    // Add empty specialized vector instances
    Py_INCREF(st->EMPTY_DOUBLE_VECTOR);
    if (PyModule_AddObject(m, "EMPTY_DOUBLE_VECTOR", st->EMPTY_DOUBLE_VECTOR) < 0) {
        Py_DECREF(st->EMPTY_DOUBLE_VECTOR);
        return -1;
    }

    Py_INCREF(st->EMPTY_LONG_VECTOR);
    if (PyModule_AddObject(m, "EMPTY_LONG_VECTOR", st->EMPTY_LONG_VECTOR) < 0) {
        Py_DECREF(st->EMPTY_LONG_VECTOR);
        return -1;
    }

    // Register types with collections.abc ABCs
    // This enables isinstance() checks for protocol compatibility
    PyObject *collections_abc = PyImport_ImportModule("collections.abc");
    if (collections_abc) {
        PyObject *result;

        // Register immutable Sequence types
        PyObject *sequence_abc = PyObject_GetAttrString(collections_abc, "Sequence");
        if (sequence_abc) {
            // Register Vector
            result = PyObject_CallMethod(sequence_abc, "register", "O", &VectorType);
            Py_XDECREF(result);

            // Register DoubleVector
            result = PyObject_CallMethod(sequence_abc, "register", "O", &DoubleVectorType);
            Py_XDECREF(result);

            // Register IntVector
            result = PyObject_CallMethod(sequence_abc, "register", "O", &IntVectorType);
            Py_XDECREF(result);

            // Register Cons as well
            result = PyObject_CallMethod(sequence_abc, "register", "O", &ConsType);
            Py_XDECREF(result);

            // Register SortedVector
            result = PyObject_CallMethod(sequence_abc, "register", "O", &SortedVectorType);
            Py_XDECREF(result);

            Py_DECREF(sequence_abc);
        }

        // Register immutable Mapping type
        PyObject *mapping_abc = PyObject_GetAttrString(collections_abc, "Mapping");
        if (mapping_abc) {
            result = PyObject_CallMethod(mapping_abc, "register", "O", &MapType);
            Py_XDECREF(result);
            Py_DECREF(mapping_abc);
        }

        // Register immutable Set type
        PyObject *set_abc = PyObject_GetAttrString(collections_abc, "Set");
        if (set_abc) {
            result = PyObject_CallMethod(set_abc, "register", "O", &SetType);
            Py_XDECREF(result);
            Py_DECREF(set_abc);
        }

        // Register TransientVector as MutableSequence
        PyObject *mutable_sequence_abc = PyObject_GetAttrString(collections_abc, "MutableSequence");
        if (mutable_sequence_abc) {
            result = PyObject_CallMethod(mutable_sequence_abc, "register", "O", &TransientVectorType);
            Py_XDECREF(result);
            Py_DECREF(mutable_sequence_abc);
        }

        // Register TransientMap as MutableMapping
        PyObject *mutable_mapping_abc = PyObject_GetAttrString(collections_abc, "MutableMapping");
        if (mutable_mapping_abc) {
            result = PyObject_CallMethod(mutable_mapping_abc, "register", "O", &TransientMapType);
            Py_XDECREF(result);
            Py_DECREF(mutable_mapping_abc);
        }

        // Register TransientSet as MutableSet
        PyObject *mutable_set_abc = PyObject_GetAttrString(collections_abc, "MutableSet");
        if (mutable_set_abc) {
            result = PyObject_CallMethod(mutable_set_abc, "register", "O", &TransientSetType);
            Py_XDECREF(result);
            Py_DECREF(mutable_set_abc);
        }

        Py_DECREF(collections_abc);
    }
    // Clear any import errors - ABC registration is optional
    PyErr_Clear();

    return 0;
}

PyMODINIT_FUNC PyInit_spork_pds(void)
{
    return PyModuleDef_Init(&pdsmodule);
}
