.. meta::
  :description: Migrate Weaviate data between different storages
  :keywords: RAG, ROCm, Weaviate, how-to, data, migration

******************************
Migrate ROCm-RAG Weaviate data
******************************

You can move data between different storages. Data migration can be approached in two ways:

- Cluster migration
- Raw data migration

Cluster migration (backup and restore)
--------------------------------------

You can move entire clusters across different storage backends. Use Weaviate's built-in backup and restore functionality to handle this data.

Raw data migration (export and import)
--------------------------------------

You can specifically move raw data objects by exporting collections to a JSON file and importing them into another machine.
 
For example, you can export one collection from the Weaviate db and save it to ``/rag-workspace/rocm-rag/backup/rocm_rag_backup_data.json`` 
on the source machine with the Weaviate instance running:

.. code:: py

  python /rag-workspace/rocm-rag/rocm_rag/data/weaviate_export_data.py

Then you can Load the ``.json`` file and import into the target machine's running Weaviate instance:

.. code:: py

  python /rag-workspace/rocm-rag/rocm_rag/data/weaviate_import_data.py

This approach is useful when you only need to migrate the dataset without transferring the cluster configuration or storage state.