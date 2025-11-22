-- Migration: Add dummy baseline metrics to model_performance_history table
-- This adds columns for tracking dummy classifier performance alongside other models

ALTER TABLE padjective.model_performance_history
ADD COLUMN IF NOT EXISTS dummy_mean_padic_loss REAL,
ADD COLUMN IF NOT EXISTS dummy_mean_accuracy REAL;
