# Cell-Type Prediction: Plain-English Data Report

## What are we trying to do?

The dataset contains measurements from **10,000 cells** taken from mouse nerve tissue. A cell is one small unit of a living body.

Every cell belongs to a **cell type**. Different cell types perform different jobs, just as people in a company may work in engineering, finance, or sales. Our task is to look at the information recorded for each cell and predict its type.

The organizers give us:

- **5,000 training cells:** their correct cell types are provided, so we can use them to teach a model.
- **5,000 test cells:** their cell types are hidden, and these are the cells we must predict.
- **60 possible cell types:** every prediction must be one of these 60 names.

The final score is simply the percentage of test cells classified correctly.

## What information do we have about each cell?

### Gene measurements

Genes can be thought of as named instruction sets used by cells. Different kinds of cells use different instructions, so the genes active in a cell give us clues about its type.

The dataset measures **200 genes**. For each gene, the value is the number of detected RNA molecules. In simple terms, a larger number suggests that the cell was using that gene more actively when it was measured.

These measurements are very sparse:

- The typical cell has a non-zero value for only **12 of the 200 genes**.
- The typical cell contains only **21 detected RNA molecules in total**.
- Most entries in the gene table are therefore zero.

This does not mean the data are broken. It is a normal feature of this measurement method, but it makes prediction harder because every cell provides only a small amount of information.

### Cell location

We know each cell's horizontal and vertical position in the tissue. Nearby cells can be related because tissue is organized into physical areas. Location may therefore provide useful clues.

However, location by itself is not enough. A simple method that assigns each training cell the type of its nearest neighboring cell is correct only **11.1%** of the time. That is better than the **5.6%** agreement expected from two randomly selected labels, but worse than always guessing the most common type, which would score **14.1%**.

The practical lesson is that location should be combined with gene measurements rather than used alone.

### Other information

We also have details such as:

- the mouse from which the cell came;
- the tissue section containing the cell;
- the cell's measured size;
- the position of the tissue section in the animal;
- sex of the mouse;
- region and segment identifiers for some cells.

Some fields are incomplete. `Region`, `Segment`, and `Excitatory_vs_Inhibitory` are missing for about **60% of the cells**. A model must be able to handle these missing values instead of treating them as errors.

## Are the training and test data similar?

Yes. This is good news.

- Training and test cells come from the same **10 mice**.
- They also cover the same **108 tissue sections**.
- No new metadata categories appear only in the test set.
- Gene-count distributions are very similar between training and test data.
- Training and test cells are physically mixed together inside the tissue sections.

This means the training data are likely to be useful for the test data. It also means that nearby training cells could help predict test cells.

![Train and test cells within the nine largest tissue sections](figures/spatial_train_test_split.svg)

Blue points are training cells and orange points are test cells. Their strong mixing shows that the test set was not collected from a completely separate area.

## Is every cell type equally common?

No. The dataset is highly unbalanced.

- The most common type has **703 training examples**.
- The rarest type has only **3 training examples**.
- The largest class is therefore about **234 times** larger than the smallest.

![Number of training cells for each cell type](figures/class_distribution.svg)

This matters because the model will learn common types more easily. For a type with only three examples, it is nearly impossible to learn all the ways that type can appear.

Because the competition uses overall accuracy, common classes have a larger effect on the final score. We should still examine performance for every class so that we understand where the model is failing.

## What are marker genes?

A **marker gene** is a gene that appears more often in one cell type than in other types. It works like a clue or identifying feature.

For example, if one gene is frequently active in one cell type but almost never active elsewhere, that gene can help the model recognize the type. The EDA produced a table of five possible marker genes for every cell type.

These are statistical clues, not guaranteed biological rules. They are useful for building and checking models, especially because we do not need to understand every gene's biological function to use its predictive pattern.

![Candidate marker patterns across cell types](figures/marker_heatmap.svg)

Darker squares indicate that a gene is unusually active for that cell type compared with other types.

## Important modeling warning: avoid accidental cheating

Training and test cells are mixed within the same mice and tissue sections. When we test a model locally, a completely random split may place neighboring cells in both the training and validation groups. This can make the model look better than it really is because the validation cells are very similar to cells it has already seen.

We should therefore use two checks:

1. **Random validation:** randomly hold out some cells. This is similar to the apparent train/test construction and may be closer to the leaderboard setting.
2. **Grouped validation:** hold out entire tissue sections or mice. This is a harder test of whether the model can work on a genuinely different group.

If we create features using neighboring cells, they must be rebuilt separately inside every validation split. Otherwise, hidden validation labels could accidentally influence the model. This is called **data leakage**.

## Recommended first modeling steps

1. Build a simple model using only the 200 gene measurements.
2. Compare the original counts with transformed counts that reduce the effect of unusually large values.
3. Add cell size, mouse, tissue section, position, and other metadata.
4. Represent missing values explicitly so the model knows when information was not recorded.
5. Compare a linear classification model with a tree-based model.
6. Add carefully constructed location and neighbor features.
7. Measure both overall accuracy and accuracy for each individual cell type.

The first goal should be a reliable baseline, not a complicated biological model. The dataset is small enough that clear validation and careful feature construction will probably matter more than deep-learning complexity.

## Bottom line

The data are clean and the training and test sets are well matched. The main difficulties are:

- very few measurements per cell;
- a large difference between common and rare cell types;
- missing metadata;
- the risk of overly optimistic validation because nearby cells are similar.

Gene measurements should be the main source of information. Cell location and metadata are useful supporting clues. A strong solution should combine all three while using validation that prevents accidental leakage.

