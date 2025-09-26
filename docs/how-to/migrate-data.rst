.. meta::
  :description: Migrate Weaviate data between different storages
  :keywords: RAG, ROCm, Weaviate, how-to, data, migration

******************************
Migrate ROCm-RAG Weaviate data
******************************

You can migrate data between different storage locations in a cluster through Weaviate, the AI-native vector database used by ROCm-RAG, or through local 
raw data migration.

Cluster migration (backup and restore)
======================================

You can move entire clusters across different storage backends with `Weaviate's built-in backup and restore functionality <https://weaviate.io/blog/tutorial-backup-and-restore-in-weaviate>`__.

Raw data migration (export and import)
======================================

You can specifically move raw data objects by exporting collections to a JSON file and importing them into another machine.
 
For example, you can export one collection from the Weaviate database and save it to ``/rag-workspace/rocm-rag/backup/rocm_rag_backup_data.json`` 
on the source machine with the Weaviate instance running:

.. code:: py

  python /rag-workspace/rocm-rag/rocm_rag/data/weaviate_export_data.py

Then you can load the ``.json`` file and import it into the target machine's running Weaviate instance:

.. code:: py

  python /rag-workspace/rocm-rag/rocm_rag/data/weaviate_import_data.py

This approach is helpful when you only need to migrate the dataset without transferring the cluster configuration or storage state.