"""Streamlit app for dynamic YAML configuration building."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import yaml

# Constants
SCHEMA_PATH = Path(__file__).parent.parent / "training" / "configs" / "schema.yaml"
AVAILABLE_COMMANDS = [
    {"id": "train", "name": "Train", "description": "Run main training"},
    {"id": "deploy", "name": "Deploy", "description": "Convert checkpoint to deployment artifacts"},
    {"id": "quantize", "name": "Quantize", "description": "Export checkpoint in quantized deployment format"},
    {"id": "infer", "name": "Infer", "description": "Run deployed model inference"},
    {"id": "sbert-train", "name": "SBERT Train", "description": "Train SBERT model"},
    {"id": "sbert-infer", "name": "SBERT Infer", "description": "Run SBERT inference tasks"},
]


def load_schema() -> Dict[str, Any]:
    """Load the training configuration schema."""
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def render_field(
    field_name: str,
    field_schema: Dict[str, Any],
    parent_key: str = "",
    level: int = 0,
) -> Any:
    """Render a form field based on its schema type."""
    # Get title and description from schema
    field_title = field_schema.get("title", field_name)
    field_description = field_schema.get("description", "")
    
    # Generate a unique key for the field
    field_key = f"{parent_key}.{field_name}" if parent_key else field_name

    # Handle different types
    field_type = field_schema.get("type")
    
    if field_type == "boolean":
        default = field_schema.get("examples", [False])[0]
        return st.checkbox(field_title, value=default, key=field_key, help=field_description)
    
    elif field_type == "integer":
        min_val = field_schema.get("minimum", 0)
        max_val = field_schema.get("maximum", 10000)
        examples = field_schema.get("examples", [])
        default = examples[0] if examples else min_val
        
        return st.number_input(
            field_title,
            value=int(default),
            min_value=int(min_val),
            max_value=int(max_val),
            step=1,
            key=field_key,
            help=field_description,
        )
    
    elif field_type == "number":
        min_val = field_schema.get("minimum", 0.0)
        max_val = field_schema.get("maximum", 1.0)
        examples = field_schema.get("examples", [])
        default = examples[0] if examples else min_val
        
        return st.number_input(
            field_title,
            value=float(default),
            min_value=float(min_val),
            max_value=float(max_val),
            step=0.01,
            key=field_key,
            format="%.6f",
            help=field_description,
        )
    
    elif field_type == "string":
        enum = field_schema.get("enum")
        examples = field_schema.get("examples", [])
        default = examples[0] if examples else ""
        
        if enum:
            return st.selectbox(field_title, enum, index=0 if enum else 0, key=field_key, help=field_description)
        else:
            return st.text_input(field_title, value=default, key=field_key, help=field_description)
    
    elif field_type == "array":
        items_schema = field_schema.get("items", {})
        enum = items_schema.get("enum")
        examples = field_schema.get("examples", [])
        default = examples[0] if examples else []
        
        if enum:
            st.write(f"**{field_title}**")
            if field_description:
                st.caption(field_description)
            selected = st.multiselect(
                "Select items",
                enum,
                default=default if len(default) > 0 else [enum[0]],
                key=field_key,
            )
            return selected
        else:
            st.write(f"**{field_title}**")
            if field_description:
                st.caption(field_description)
            array_input = st.text_area(
                "Enter items (one per line)",
                value="\n".join(map(str, default)),
                key=field_key,
            )
            return [line.strip() for line in array_input.split("\n") if line.strip()]
    
    elif field_type == "object":
        st.write(f"### {field_title}")
        if field_description:
            st.caption(field_description)
        return render_object(field_name, field_schema, parent_key, level + 1)
    
    return None


def render_object(
    obj_name: str,
    obj_schema: Dict[str, Any],
    parent_key: str = "",
    level: int = 0,
) -> Dict[str, Any]:
    """Render an object schema with all its properties."""
    properties = obj_schema.get("properties", {})
    required = obj_schema.get("required", [])
    
    result = {}
    parent_key = f"{parent_key}.{obj_name}" if parent_key else obj_name
    
    for prop_name, prop_schema in properties.items():
        with st.container():
            # Get title from schema for display
            prop_title = prop_schema.get("title", prop_name)
            prop_description = prop_schema.get("description", "")
            
            # Mark required fields
            is_required = prop_name in required
            if is_required:
                st.markdown(f"**{prop_title}** *:red[Required]*")
            else:
                st.markdown(f"**{prop_title}**")
            
            # Show description if available
            if prop_description:
                st.caption(prop_description)
            
            prop_value = render_field(prop_name, prop_schema, parent_key, level)
            
            if prop_value is not None:
                result[prop_name] = prop_value
            
            st.divider()
    
    return result


def render_optimizer_section(optimizer_class: str) -> Dict[str, Any]:
    """Render the optimizer configuration section."""
    st.header("Optimizer Configuration")
    
    result = {
        "optimizer_class": optimizer_class,
    }
    
    # Load schema for optimizer parameters
    schema = load_schema()
    optimizer_schema = schema["properties"]["training"]["properties"]["optimizer"]
    optimizer_class_schema = optimizer_schema["properties"]["optimizer_class"]
    
    st.subheader("Optimizer Parameters")
    st.info(
        "The following parameters are available for this optimizer. "
        "Prefix each parameter with the optimizer class name (e.g., 'adamw-lr_embeddings')."
    )
    
    # Map optimizer class to its prefix
    prefix_map = {
        "sgd_momentum": "sgd_momentum",
        "adamw": "adamw",
        "adafactor": "adafactor",
        "galore_adamw": "galore_adamw",
        "prodigy": "prodigy",
        "lion": "lion",
        "sophia": "sophia",
        "muon": "muon",
        "turbo_muon": "turbo_muon",
        "radam": "radam",
        "adan": "adan",
        "adopt": "adopt",
        "ademamix": "ademamix",
        "mars_adamw": "mars_adamw",
        "cautious_adamw": "cautious_adamw",
        "lamb": "lamb",
        "schedulefree_adamw": "schedulefree_adamw",
        "shampoo": "shampoo",
        "soap": "soap",
    }
    
    prefix = prefix_map.get(optimizer_class, optimizer_class)
    
    # Render common parameter groups
    param_groups = {
        "embeddings": st.container(),
        "norms": st.container(),
        "ode": st.container(),
        "retnet": st.container(),
        "mamba": st.container(),
        "attention": st.container(),
        "other": st.container(),
    }
    
    # Descriptions for parameter groups
    param_descriptions = {
        "lr_": "Learning rate controls step size for parameter updates. Higher values converge faster but may overshoot. Lower values are more stable but slower.",
        "wd_": "Weight decay regularizes by penalizing large weights. Higher values prevent overfitting but may underfit.",
        "betas_": "Beta coefficients for momentum (β₁) and squared gradient (β₂). Controls momentum strength and second moment.",
        "eps_": "Epsilon prevents division by zero in adaptive optimizers. Small values ensure numerical stability.",
    }
    
    with param_groups["embeddings"]:
        st.write("##### Embeddings")
        st.caption("Parameters controlling token embedding matrix optimization.")
        lr_emb = st.number_input(
            "Learning Rate",
            value=1e-6,
            min_value=0.0,
            format="%.1e",
            key=f"{prefix}-lr_embeddings",
            help=f"{prefix}-lr_embeddings: {param_descriptions['lr_']}",
        )
        wd_emb = st.number_input(
            "Weight Decay",
            value=0.01,
            min_value=0.0,
            format="%.3f",
            key=f"{prefix}-wd_embeddings",
            help=f"{prefix}-wd_embeddings: {param_descriptions['wd_']}",
        )
        result[f"{prefix}-lr_embeddings"] = lr_emb
        result[f"{prefix}-wd_embeddings"] = wd_emb
    
    with param_groups["norms"]:
        st.write("##### Normalization Layers")
        st.caption("Parameters for layer normalization and other normalization layers.")
        lr_norm = st.number_input(
            "Learning Rate",
            value=5e-6,
            min_value=0.0,
            format="%.1e",
            key=f"{prefix}-lr_norms",
            help=f"{prefix}-lr_norms: {param_descriptions['lr_']}",
        )
        wd_norm = st.number_input(
            "Weight Decay",
            value=0.001,
            min_value=0.0,
            format="%.4f",
            key=f"{prefix}-wd_norms",
            help=f"{prefix}-wd_norms: {param_descriptions['wd_']}",
        )
        result[f"{prefix}-lr_norms"] = lr_norm
        result[f"{prefix}-wd_norms"] = wd_norm
    
    with param_groups["attention"]:
        st.write("##### Attention Layers")
        st.caption("Parameters for attention mechanism optimization (all attention variants).")
        lr_attn = st.number_input(
            "Learning Rate",
            value=3e-6,
            min_value=0.0,
            format="%.1e",
            key=f"{prefix}-lr_attention",
            help=f"{prefix}-lr_attention: {param_descriptions['lr_']}",
        )
        wd_attn = st.number_input(
            "Weight Decay",
            value=0.01,
            min_value=0.0,
            format="%.3f",
            key=f"{prefix}-wd_attention",
            help=f"{prefix}-wd_attention: {param_descriptions['wd_']}",
        )
        result[f"{prefix}-lr_attention"] = lr_attn
        result[f"{prefix}-wd_attention"] = wd_attn
    
    with param_groups["other"]:
        st.write("##### Other Parameters")
        st.caption("Parameters for remaining model components (FFN, routing, etc.).")
        lr_other = st.number_input(
            "Learning Rate",
            value=2e-6,
            min_value=0.0,
            format="%.1e",
            key=f"{prefix}-lr_other",
            help=f"{prefix}-lr_other: {param_descriptions['lr_']}",
        )
        wd_other = st.number_input(
            "Weight Decay",
            value=0.01,
            min_value=0.0,
            format="%.3f",
            key=f"{prefix}-wd_other",
            help=f"{prefix}-wd_other: {param_descriptions['wd_']}",
        )
        betas_other = st.text_input(
            "Betas",
            value="[0.9, 0.95]",
            key=f"{prefix}-betas_other",
            help=f"{prefix}-betas_other: {param_descriptions['betas_']}",
        )
        eps_other = st.number_input(
            "Epsilon",
            value=1e-8,
            min_value=0.0,
            format="%.1e",
            key=f"{prefix}-eps_other",
            help=f"{prefix}-eps_other: {param_descriptions['eps_']}",
        )
        result[f"{prefix}-lr_other"] = lr_other
        result[f"{prefix}-wd_other"] = wd_other
        result[f"{prefix}-betas_other"] = betas_other
        result[f"{prefix}-eps_other"] = eps_other
    
    return result


def build_config_from_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Build the configuration dictionary from the schema and user input."""
    config = {}
    
    # Model configuration
    st.header("Model Configuration")
    st.subheader("Choose Model Type")
    model_mode = st.radio(
        "Model Mode",
        ["Custom Model", "Base Model"],
        horizontal=True,
    )
    
    if model_mode == "Custom Model":
        with st.expander("Model Parameters", expanded=True):
            model_config = render_object("model", schema["properties"]["model"])
            config["model_class"] = st.selectbox(
                "Model Class",
                schema["properties"]["model_class"]["enum"],
                index=0,
                key="model_class",
            )
            config["model"] = model_config
    else:
        with st.expander("Base Model Configuration", expanded=True):
            base_model = st.text_input(
                "Base Model ID/Path",
                value="answerdotai/ModernBERT-base",
                key="base_model",
            )
            config["base_model"] = base_model
            
            st.write("**Tokenizer Configuration**")
            tokenizer_config = render_object("tokenizer", schema["properties"]["tokenizer"])
            config["tokenizer"] = tokenizer_config
    
    # Training configuration
    st.header("Training Configuration")
    with st.expander("Training Parameters", expanded=True):
        training_schema = schema["properties"]["training"]
        
        # Task selection
        task = st.selectbox(
            "Training Task",
            training_schema["properties"]["task"]["enum"],
            index=0,
            key="training.task",
        )
        
        training_config = {"task": task}
        
        # General training parameters
        st.write("##### General Training Parameters")
        num_epochs_schema = training_schema["properties"]["num_epochs"]
        batch_size_schema = training_schema["properties"]["batch_size"]
        max_length_schema = training_schema["properties"]["max_length"]
        
        num_epochs = st.number_input(
            num_epochs_schema.get("title", "Number of Epochs"),
            value=5,
            min_value=1,
            key="training.num_epochs",
            help=num_epochs_schema.get("description", ""),
        )
        batch_size = st.number_input(
            batch_size_schema.get("title", "Batch Size"),
            value=4,
            min_value=1,
            key="training.batch_size",
            help=batch_size_schema.get("description", ""),
        )
        max_length = st.number_input(
            max_length_schema.get("title", "Max Sequence Length"),
            value=512,
            min_value=1,
            key="training.max_length",
            help=max_length_schema.get("description", ""),
        )
        
        training_config["num_epochs"] = num_epochs
        training_config["batch_size"] = batch_size
        training_config["max_length"] = max_length
        
        # Dataset parameters
        st.write("##### Dataset Parameters")
        max_samples_schema = training_schema["properties"]["max_samples"]
        dataset_batch_size_schema = training_schema["properties"]["dataset_batch_size"]
        
        max_samples = st.number_input(
            max_samples_schema.get("title", "Max Samples"),
            value=20000000,
            min_value=1,
            key="training.max_samples",
            help=max_samples_schema.get("description", ""),
        )
        dataset_batch_size = st.number_input(
            dataset_batch_size_schema.get("title", "Dataset Batch Size"),
            value=25000,
            min_value=1,
            key="training.dataset_batch_size",
            help=dataset_batch_size_schema.get("description", ""),
        )
        
        training_config["max_samples"] = max_samples
        training_config["dataset_batch_size"] = dataset_batch_size
        
        # Optimizer
        st.write("##### Optimizer")
        optimizer_class = st.selectbox(
            "Optimizer Class",
            schema["properties"]["training"]["properties"]["optimizer"]["properties"]["optimizer_class"]["enum"],
            index=1,  # Default to adamw
            key="training.optimizer.optimizer_class",
        )
        
        optimizer_params = render_optimizer_section(optimizer_class)
        training_config["optimizer"] = optimizer_params
        
        # Scheduler
        st.write("##### Scheduler")
        scheduler_total_steps_schema = training_schema["properties"]["scheduler_total_steps"]
        scheduler_warmup_ratio_schema = training_schema["properties"]["scheduler_warmup_ratio"]
        scheduler_type_schema = training_schema["properties"]["scheduler_type"]
        
        scheduler_total_steps = st.number_input(
            scheduler_total_steps_schema.get("title", "Total Steps"),
            value=10000,
            min_value=1,
            key="training.scheduler_total_steps",
            help=scheduler_total_steps_schema.get("description", ""),
        )
        scheduler_warmup_ratio = st.number_input(
            scheduler_warmup_ratio_schema.get("title", "Warmup Ratio"),
            value=0.1,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key="training.scheduler_warmup_ratio",
            help=scheduler_warmup_ratio_schema.get("description", ""),
        )
        scheduler_type = st.selectbox(
            scheduler_type_schema.get("title", "Scheduler Type"),
            scheduler_type_schema.get("enum", []),
            index=0,
            key="training.scheduler_type",
            help=scheduler_type_schema.get("description", ""),
        )
        
        training_config["scheduler_total_steps"] = scheduler_total_steps
        training_config["scheduler_warmup_ratio"] = scheduler_warmup_ratio
        training_config["scheduler_type"] = scheduler_type
        
        # Gradient settings
        st.write("##### Gradient Settings")
        gradient_accumulation_steps_schema = training_schema["properties"]["gradient_accumulation_steps"]
        grad_clip_max_norm_schema = training_schema["properties"]["grad_clip_max_norm"]
        
        gradient_accumulation_steps = st.number_input(
            gradient_accumulation_steps_schema.get("title", "Gradient Accumulation Steps"),
            value=4,
            min_value=1,
            key="training.gradient_accumulation_steps",
            help=gradient_accumulation_steps_schema.get("description", ""),
        )
        grad_clip_max_norm = st.number_input(
            grad_clip_max_norm_schema.get("title", "Gradient Clip Max Norm"),
            value=5.0,
            min_value=0.0,
            key="training.grad_clip_max_norm",
            help=grad_clip_max_norm_schema.get("description", ""),
        )
        
        training_config["gradient_accumulation_steps"] = gradient_accumulation_steps
        training_config["grad_clip_max_norm"] = grad_clip_max_norm
        
        # Checkpoint settings
        st.write("##### Checkpoint Settings")
        checkpoint_every_n_steps_schema = training_schema["properties"]["checkpoint_every_n_steps"]
        max_rolling_checkpoints_schema = training_schema["properties"]["max_rolling_checkpoints"]
        num_best_checkpoints_schema = training_schema["properties"]["num_best_checkpoints"]
        
        checkpoint_every_n_steps = st.number_input(
            checkpoint_every_n_steps_schema.get("title", "Checkpoint Every N Steps"),
            value=500,
            min_value=1,
            key="training.checkpoint_every_n_steps",
            help=checkpoint_every_n_steps_schema.get("description", ""),
        )
        max_rolling_checkpoints = st.number_input(
            max_rolling_checkpoints_schema.get("title", "Max Rolling Checkpoints"),
            value=3,
            min_value=1,
            key="training.max_rolling_checkpoints",
            help=max_rolling_checkpoints_schema.get("description", ""),
        )
        num_best_checkpoints = st.number_input(
            num_best_checkpoints_schema.get("title", "Num Best Checkpoints"),
            value=2,
            min_value=1,
            key="training.num_best_checkpoints",
            help=num_best_checkpoints_schema.get("description", ""),
        )
        
        training_config["checkpoint_every_n_steps"] = checkpoint_every_n_steps
        training_config["max_rolling_checkpoints"] = max_rolling_checkpoints
        training_config["num_best_checkpoints"] = num_best_checkpoints
        
        # Logging settings
        st.write("##### Logging Settings")
        csv_log_path_schema = training_schema["properties"]["csv_log_path"]
        log_gradient_stats_schema = training_schema["properties"]["log_gradient_stats"]
        gradient_log_interval_schema = training_schema["properties"]["gradient_log_interval"]
        
        csv_log_path = st.text_input(
            csv_log_path_schema.get("title", "CSV Log Path"),
            value="training_metrics.csv",
            key="training.csv_log_path",
            help=csv_log_path_schema.get("description", ""),
        )
        log_gradient_stats = st.checkbox(
            log_gradient_stats_schema.get("title", "Log Gradient Stats"),
            value=True,
            key="training.log_gradient_stats",
            help=log_gradient_stats_schema.get("description", ""),
        )
        gradient_log_interval = st.number_input(
            gradient_log_interval_schema.get("title", "Gradient Log Interval"),
            value=10,
            min_value=1,
            key="training.gradient_log_interval",
            help=gradient_log_interval_schema.get("description", ""),
        )
        
        training_config["csv_log_path"] = csv_log_path
        training_config["log_gradient_stats"] = log_gradient_stats
        training_config["gradient_log_interval"] = gradient_log_interval
        
        # GPU settings
        st.write("##### GPU Settings")
        gpu_temp_guard_enabled_schema = training_schema["properties"]["gpu_temp_guard_enabled"]
        gpu_temp_pause_threshold_c_schema = training_schema["properties"]["gpu_temp_pause_threshold_c"]
        gpu_temp_resume_threshold_c_schema = training_schema["properties"]["gpu_temp_resume_threshold_c"]
        
        gpu_temp_guard_enabled = st.checkbox(
            gpu_temp_guard_enabled_schema.get("title", "Enable GPU Temperature Guard"),
            value=True,
            key="training.gpu_temp_guard_enabled",
            help=gpu_temp_guard_enabled_schema.get("description", ""),
        )
        gpu_temp_pause_threshold_c = st.number_input(
            gpu_temp_pause_threshold_c_schema.get("title", "GPU Temp Pause Threshold (°C)"),
            value=90.0,
            min_value=0.0,
            key="training.gpu_temp_pause_threshold_c",
            help=gpu_temp_pause_threshold_c_schema.get("description", ""),
        )
        gpu_temp_resume_threshold_c = st.number_input(
            gpu_temp_resume_threshold_c_schema.get("title", "GPU Temp Resume Threshold (°C)"),
            value=80.0,
            min_value=0.0,
            key="training.gpu_temp_resume_threshold_c",
            help=gpu_temp_resume_threshold_c_schema.get("description", ""),
        )
        
        training_config["gpu_temp_guard_enabled"] = gpu_temp_guard_enabled
        training_config["gpu_temp_pause_threshold_c"] = gpu_temp_pause_threshold_c
        training_config["gpu_temp_resume_threshold_c"] = gpu_temp_resume_threshold_c
        
        # Mixed precision
        use_amp_schema = training_schema["properties"]["use_amp"]
        use_amp = st.checkbox(
            use_amp_schema.get("title", "Use Automatic Mixed Precision"),
            value=False,
            key="training.use_amp",
            help=use_amp_schema.get("description", ""),
        )
        training_config["use_amp"] = use_amp
        
        config["training"] = training_config
    
    return config


def build_cli_command(command: str, config: Dict[str, Any], output_path: str) -> str:
    """Build the CLI command based on the selected command and configuration."""
    cmd_parts = ["python", "-m", "src.cli", command]
    
    if command == "train":
        cmd_parts.extend(["--config", output_path])
        cmd_parts.extend(["--config-name", config.get("model_class", "mini")])
    elif command in ["deploy", "quantize"]:
        cmd_parts.extend(["--config", output_path])
    elif command == "infer":
        cmd_parts.extend(["--config", output_path])
    elif command == "sbert-train":
        cmd_parts.extend(["--config", output_path])
    elif command == "sbert-infer":
        cmd_parts.extend(["--config", output_path])
    
    return " ".join(str(part) for part in cmd_parts)


def run_command_with_nohup(command: str, log_file: str) -> subprocess.Popen:
    """Run a command with nohup in the background."""
    nohup_cmd = f"nohup {command} > {log_file} 2>&1 &"
    process = subprocess.Popen(
        nohup_cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return process


def main(argv=None):
    """Main Streamlit app entry point."""
    st.set_page_config(
        page_title="Transformer Encoder Frankenstein - Config Builder",
        page_icon="🤖",
        layout="wide",
    )
    
    st.title("🤖 Transformer Encoder Frankenstein - Config Builder")
    st.markdown("Build YAML configuration files and generate CLI commands dynamically.")
    
    # Load schema
    try:
        schema = load_schema()
        st.success("Schema loaded successfully!")
    except Exception as e:
        st.error(f"Error loading schema: {e}")
        return
    
    # Sidebar for command selection
    st.sidebar.header("Command Selection")
    command_info = st.sidebar.selectbox(
        "Select Command",
        AVAILABLE_COMMANDS,
        format_func=lambda x: f"{x['name']}: {x['description']}",
    )
    
    command_id = command_info["id"]
    st.sidebar.info(f"Selected: **{command_info['name']}**\n\n{command_info['description']}")
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["Configuration Builder", "YAML Output", "Command Execution"])
    
    with tab1:
        st.header("Configuration Builder")
        st.info("Fill in the form below to build your configuration.")
        
        # Build configuration from schema
        config = build_config_from_schema(schema)
        
        # Store config in session state
        st.session_state["config"] = config
    
    with tab2:
        st.header("YAML Output")
        
        if "config" in st.session_state:
            config = st.session_state["config"]
            yaml_output = yaml.dump(config, default_flow_style=False, sort_keys=False)
            
            st.subheader("Generated YAML Configuration")
            st.code(yaml_output, language="yaml")
            
            # Download button
            st.download_button(
                label="Download YAML File",
                data=yaml_output,
                file_name="config.yaml",
                mime="text/yaml",
            )
            
            # Save path input
            output_path = st.text_input(
                "Save Configuration To",
                value="./config_generated.yaml",
                key="output_path",
            )
            
            if st.button("Save to File"):
                try:
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(yaml_output)
                    st.success(f"Configuration saved to {output_path}")
                except Exception as e:
                    st.error(f"Error saving file: {e}")
        else:
            st.warning("Please go to the 'Configuration Builder' tab to create a configuration first.")
    
    with tab3:
        st.header("Command Execution")
        
        if "config" in st.session_state:
            config = st.session_state["config"]
            output_path = st.text_input(
                "Configuration File Path",
                value="./config_generated.yaml",
                key="exec_output_path",
            )
            
            # Build CLI command
            cli_command = build_cli_command(command_id, config, output_path)
            
            st.subheader("Generated CLI Command")
            st.code(cli_command, language="bash")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📋 Copy to Clipboard"):
                    st.code_area("Copy this command:", cli_command, height=100, key="copy_command")
                    st.success("Command displayed above - copy it manually")
            
            with col2:
                st.write("### Run with nohup")
                log_file = st.text_input(
                    "Log File Path",
                    value="./nohup_web.out",
                    key="log_file",
                )
                
                if st.button("▶️ Run with nohup"):
                    try:
                        process = run_command_with_nohup(cli_command, log_file)
                        st.success(f"Command started with nohup. Check logs at: {log_file}")
                        st.info(f"Process ID: {process.pid}")
                    except Exception as e:
                        st.error(f"Error running command: {e}")
            
            # Command options
            st.subheader("Command Options")
            show_full_command = st.checkbox("Show Full Command", value=False, key="show_full")
            
            if show_full_command:
                full_command = f"nohup {cli_command} > {st.session_state.get('log_file', './nohup_web.out')} 2>&1 &"
                st.code(full_command, language="bash")
        else:
            st.warning("Please go to the 'Configuration Builder' tab to create a configuration first.")


if __name__ == "__main__":
    main()
