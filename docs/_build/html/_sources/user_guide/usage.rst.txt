Usage Guide
===========

Full workflow for evaluating and enhancing dataset descriptions.

Main Workflow
-------------

1. **Enter Description**
   - Paste your dataset description in the text area
   - Can be a few sentences or multiple paragraphs

2. **Evaluate**
   - Click "Evaluate Description"
   - View rubric breakdown: 10 criteria × 1 point each = 0–10 score
   - Each criterion shows:
     - ✓ (passed) or ✗ (not met)
     - Explanation of what's missing or good

3. **Upload Context Documents** (optional)
   - Click "Upload Files"
   - Select PDFs or DOCX files (papers, reports, technical docs)
   - Rocco will chunk and embed them for RAG

4. **Enhance Description**
   - Write specific feedback in the feedback text area
   - Examples:
     - "Add sample preparation details"
     - "Explain the imaging method"
     - "Include permeability data if available"
   - Click "Enhance with Rocco"
   - Rocco validates your feedback, then improves the description

5. **Review Results**
   - See the enhanced description side-by-side with the original
   - Check citations (each added statement shows its source)
   - Accept, reject, or manually edit

6. **Iterate**
   - Click "Enhance with Rocco" again for further refinement
   - Save your session to resume later

Evaluation Rubric
-----------------

Rocco scores descriptions on 10 criteria (1 point each, total 0–10):

1. **Self-Contained Description**
   - Does it stand alone without external context?

2. **Methodology Clarity**
   - Is the imaging/measurement technique explained?

3. **Data Organization**
   - Is the structure clear (e.g., "images in TIFF stacks")?

4. **Sample Characteristics**
   - Are materials described (rock type, porosity, etc.)?

5. **Spatial Resolution**
   - Are voxel size or pixel dimensions stated?

6. **Quality Control**
   - Are there checks for data validity (alignment, artifacts, etc.)?

7. **Accessibility**
   - Are file formats and access instructions clear?

8. **Data Completeness**
   - Is the scope of the dataset covered (# samples, coverage)?

9. **Standards Compliance**
   - Are relevant standards or best practices mentioned?

10. **Completeness of Metadata**
    - Are key metadata fields provided (DOI, license, creators)?

Score Interpretation
~~~~~~~~~~~~~~~~~~~~

- **8–10**: Excellent. Clear, complete, citations ready.
- **6–7**: Good. Some details missing or unclear.
- **4–5**: Fair. Major gaps in methodology or scope.
- **0–3**: Poor. Very incomplete or vague.

Session Management
------------------

**Saving Sessions**

Your session is automatically saved after each action:
- Description text
- Evaluation results
- Uploaded documents and vector index
- Conversation history
- All refinements

Sessions are stored in timestamped JSON files in the `sessions/` directory (if configured).

**Resuming a Session**

*(Future feature)* Load a previous session by ID to continue refinement where you left off.

**Exporting**

*(Future feature)* Export the final description, evaluation report, and citations in multiple formats:
- Markdown
- JSON (structured data)
- PDF (formatted report)

Citations
---------

When Rocco enhances your description, each new statement is traced to its source:

.. code-block:: json

   {
     "statement": "The samples were prepared following ASTM standards for core handling.",
     "source": "context_chunk",
     "quote": "...standard core handling protocols (standardized in ASTM standards)...",
     "doc_title": "Smith_et_al_2015.pdf",
     "page": 3,
     "chunk_index": 5
   }

**Source types:**
- ``original_description`` — came from your original text
- ``context_chunk`` — from an uploaded document
- ``user_feedback`` — based on your feedback

Use citations to verify claims and credit sources.

Advanced Features
-----------------

**Content Screening**

Before enhancing, Rocco validates your feedback:

- **Relevant?** Does it pertain to the dataset?
- **Accurate?** Is it consistent with existing data?
- **Respectful?** Is the tone professional?
- **Coherent?** Does it make sense?

Feedback marked as **"Flag for Review"** will show a human-in-the-loop UI for verification.

**Multi-Turn Refinement**

You can enhance multiple times:

1. Initial description → Evaluate → Feedback → Enhance (Round 1)
2. Enhanced description → Evaluate → Feedback → Enhance (Round 2)
3. ... repeat as needed

Each round includes full conversation history so Rocco understands context.

**RAG with Multiple Documents**

Upload multiple context documents:

- Rocco will blend results from all documents
- More documents → stronger context signal
- But also: potential for conflicting information (Rocco will note this)

Tips & Best Practices
---------------------

✓ **DO:**
   - Provide complete, honest feedback
   - Upload relevant, high-quality documents
   - Iterate multiple times for polish
   - Review citations to verify accuracy
   - Use the rubric as a checklist before writing initial descriptions

✗ **DON'T:**
   - Ask Rocco to invent missing data (it won't)
   - Expect perfection in one pass
   - Ignore citations or sources
   - Upload irrelevant documents (adds noise)
   - Overwrite all feedback with manual edits (defeats the purpose)

Common Use Cases
----------------

**Use Case 1: Quick Evaluation**

Time: ~2 minutes

.. code-block:: text

   1. Paste description
   2. Click "Evaluate"
   3. See score and gaps

Good for: Quick feedback on a description you've written.

**Use Case 2: Full Enhancement**

Time: ~10 minutes

.. code-block:: text

   1. Paste description
   2. Evaluate
   3. Upload 1–2 relevant papers
   4. Write detailed feedback
   5. Enhance
   6. Review and accept

Good for: Improving descriptions for publication.

**Use Case 3: Iterative Refinement**

Time: ~30 minutes

.. code-block:: text

   1. Paste description (initial quality: ~5/10)
   2. Evaluate, enhance (→ 6/10)
   3. Add more feedback, enhance (→ 7/10)
   4. Fine-tune wording, enhance (→ 8/10)
   5. Accept final

Good for: Preparing high-quality descriptions for archival.

Troubleshooting
---------------

**"Enhancement failed"**
   - Check that you set ``LLM_API_KEY`` in ``.env``
   - Verify your API key is valid and has sufficient quota
   - Check your internet connection

**"No context found"**
   - Ensure documents were uploaded successfully
   - Check that documents are PDFs or DOCX (other formats not supported)
   - Try uploading different documents

**"Feedback marked as 'Reject'"**
   - Rewrite your feedback more clearly or concisely
   - Ensure feedback is relevant to the dataset
   - Try splitting long feedback into multiple smaller suggestions

Next Steps
==========

- Explore :doc:`configuration` to switch LLM providers
- Read :doc:`../developer_guide/architecture` for how the system works
- Check :doc:`../developer_guide/contributing` to contribute improvements
