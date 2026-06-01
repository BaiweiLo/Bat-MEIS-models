#install.packages('phylolm')

# Load packages
library(phylolm)
library(ape)
library(dplyr)
library(tidyr)

# ---- Load predictions of Laurasiatherians ----
expr <- read.table(
  "example_meis_binding_prediction.tsv",
  header = TRUE, sep = "\t", check.names = FALSE
)
# ---- Load Zoonomia newick tree (can be found on their website) ----
tree <- read.tree("241-mammalian-2020v2.phast-242.nh")

# ---- Define species columns ----
# ---- Bats ----
order1_cols <- 9:38
# ---- Non-bat Scrotiferans -----
order2_cols <- 39:ncol(expr)
order1_species <- colnames(expr)[order1_cols]
order2_species <- colnames(expr)[order2_cols]

# ---- Filter genes with ≥15 species per group ----
expr_filtered <- expr %>%
  rowwise() %>%
  mutate(
    n_order1 = sum(!is.na(c_across(all_of(order1_species)))),
    n_order2 = sum(!is.na(c_across(all_of(order2_species))))
  ) %>%
  ungroup() %>%
  filter(n_order1 >= 15, n_order2 >= 15)

# ---- Reshape to long format ----
expr_long <- expr_filtered %>%
  pivot_longer(
    cols = -c(Gene, n_order1, n_order2),
    names_to = "species",
    values_to = "expression"
  ) %>%
  mutate(order = case_when(
    species %in% order1_species ~ "Order1",
    species %in% order2_species ~ "Order2",
    TRUE ~ NA_character_
  )) %>%
  filter(!is.na(order), !is.na(expression)) # reduce memory early

# ---- Predefine output (avoid growing list dynamically) ----
genes <- expr_filtered$Gene
n_genes <- length(genes)
results <- data.frame(Gene = genes, P_Value_OrderOrder2 = NA_real_)

# ---- Main loop ----
for (i in seq_along(genes)) {
  g <- genes[i]
  if (i %% 50 == 0) message("Processing gene ", i, "/", n_genes, ": ", g)

  tryCatch({
    df <- expr_long[expr_long$Gene == g, c("species", "order", "expression")]
    sp_in_data <- intersect(df$species, tree$tip.label)

    # Skip if too few species
    if (length(sp_in_data) < 3) stop("Not enough species after filtering")

    sub_tree <- keep.tip(tree, sp_in_data)
    df <- df[df$species %in% sp_in_data, ]
    df <- df[match(sub_tree$tip.label, df$species), ]
    rownames(df) <- df$species

    fit <- phylolm(expression ~ order, data = df, phy = sub_tree, model = "lambda")
    coef_table <- summary(fit)$coefficients

    if ("orderOrder2" %in% rownames(coef_table)) {
      results$P_Value_OrderOrder2[i] <- coef_table["orderOrder2", "p.value"]
    }

    rm(df, sub_tree, fit, coef_table) # free memory
    gc(verbose = FALSE)

  }, error = function(e) {
    results$P_Value_OrderOrder2[i] <- NA_real_
  })
}

# ---- Add FDR correction ----
results$FDR <- p.adjust(results$P_Value_OrderOrder2, method = "fdr")

# ---- Save ----
write.table(results, "meis_phylolm.tsv",
            sep = "\t", quote = FALSE, row.names = FALSE)
