import streamlit as st
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from PIL import Image
import io
import random

# Set page config
st.set_page_config(
    page_title="Multimodal AI - Image Caption Generator",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stSidebar {
        background-color: #262730;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #262730;
        border-radius: 4px 4px 0px 0px;
        gap: 5px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f2937;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #374151;
    }
    .stMetric {
        background-color: #1f2937;
        border: 1px solid #374151;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stSuccess {
        background-color: #064e3b;
        border: 1px solid #10b981;
    }
    .stInfo {
        background-color: #1e3a8a;
        border: 1px solid #3b82f6;
    }
    .stWarning {
        background-color: #78350f;
        border: 1px solid #f59e0b;
    }
    .stError {
        background-color: #7f1d1d;
        border: 1px solid #ef4444;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🖼️ Multimodal AI - Image Caption Generator")

# Sidebar for model selection and info
with st.sidebar:
    st.header("Model Selection")
    
    # Model selection radio button
    model_choice = st.radio(
        "Choose BLIP Model:",
        ["Base Model", "Large Model"],
        help="Base model is faster but less accurate. Large model is more accurate but slower."
    )
    
    # Map selection to model names
    model_mapping = {
        "Base Model": "Salesforce/blip-image-captioning-base",
        "Large Model": "Salesforce/blip-image-captioning-large"
    }
    
    selected_model = model_mapping[model_choice]
    
    st.markdown("---")
    st.header("Model Information")
    st.info(f"Using: {selected_model}")
    
    if model_choice == "Base Model":
        st.caption("✅ Faster inference\n✅ Lower memory usage\n⚠️ Less detailed captions")
    else:
        st.caption("✅ More detailed captions\n✅ Better accuracy\n⚠️ Slower inference\n⚠️ Higher memory usage")

# Initialize session state for model loading
@st.cache_resource(show_spinner="Loading BLIP model...")
def load_model(model_name):
    """Load the BLIP model and processor with improved caching"""
    try:
        with st.spinner(f"Loading {model_name}..."):
            processor = BlipProcessor.from_pretrained(
                model_name,
                cache_dir="./model_cache",
                local_files_only=False
            )
            model = BlipForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir="./model_cache",
                local_files_only=False,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            return processor, model
    except Exception as e:
        st.error(f"❌ Error loading model {model_name}: {str(e)}")
        return None, None

@st.cache_resource(show_spinner="Loading text generator...")
def load_text_generator():
    """Load the GPT-2 text generation pipeline for caption enhancement with improved caching"""
    try:
        with st.spinner("Loading GPT-2 text generator..."):
            text_generator = pipeline(
                "text-generation", 
                model="gpt2", 
                max_length=100, 
                do_sample=True,
                cache_dir="./model_cache",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            return text_generator
    except Exception as e:
        st.error(f"❌ Error loading text generator: {str(e)}")
        return None

# Load selected model
processor, model = load_model(selected_model)

if processor is None or model is None:
    st.error("❌ Failed to load the model. Please check your internet connection and try again.")
    st.stop()

# Load text generator for caption enhancement
text_generator = load_text_generator()

if text_generator is None:
    st.warning("⚠️ Text generator failed to load. Caption enhancement will not be available.")

def enhance_caption(original_caption, text_generator):
    """Generate enhanced variations of the original caption using GPT-2"""
    if text_generator is None:
        return []
    
    try:
        # Create different prompts to generate variations
        prompts = [
            f"Describe this image in detail: {original_caption}",
            f"Write a creative caption for this image: {original_caption}",
            f"Create an artistic description: {original_caption}"
        ]
        
        enhanced_captions = []
        
        for prompt in prompts[:2]:  # Generate 2 variations
            # Generate text with different parameters for variety
            result = text_generator(
                prompt,
                max_length=len(prompt.split()) + 20,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                pad_token_id=text_generator.tokenizer.eos_token_id
            )
            
            # Extract the generated text and clean it up
            generated_text = result[0]['generated_text']
            # Remove the original prompt to get just the enhancement
            enhanced_text = generated_text.replace(prompt, "").strip()
            
            # Clean up the text
            if enhanced_text and len(enhanced_text) > 10:
                # Remove any incomplete sentences at the end
                sentences = enhanced_text.split('.')
                if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
                    enhanced_text = '.'.join(sentences[:-1]) + '.'
                
                enhanced_captions.append(enhanced_text)
        
        return enhanced_captions[:2]  # Return up to 2 enhanced captions
        
    except Exception as e:
        st.error(f"Error enhancing caption: {str(e)}")
        return []

# Initialize session state for storing captions
if 'original_caption' not in st.session_state:
    st.session_state.original_caption = None
if 'enhanced_captions' not in st.session_state:
    st.session_state.enhanced_captions = []
if 'current_image' not in st.session_state:
    st.session_state.current_image = None

# File uploader with error handling
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=['jpg', 'jpeg', 'png'],
    help="Upload an image to generate a caption"
)

# Process uploaded image with error handling
if uploaded_file is not None:
    try:
        st.session_state.current_image = Image.open(uploaded_file)
        # Clear previous captions when new image is uploaded
        st.session_state.original_caption = None
        st.session_state.enhanced_captions = []
        st.success("✅ Image uploaded successfully!")
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        st.error("Please try uploading a different image file.")
        st.session_state.current_image = None

# Create tabs
if st.session_state.current_image is not None:
    tab1, tab2, tab3 = st.tabs(["📸 Image & Caption", "✨ Enhanced Captions", "📊 Stats"])
    
    with tab1:
        # Image & Caption Tab
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Uploaded Image")
            st.image(st.session_state.current_image, caption="Your uploaded image", use_column_width=True)
        
        with col2:
            st.subheader("🤖 Generated Caption")
            
            # Show current model being used
            st.info(f"Using: **{model_choice}** ({selected_model})")
            
            # Generate caption button
            if st.button("Generate Caption", type="primary"):
                with st.spinner(f"Generating caption using {model_choice}..."):
                    try:
                        # Process the image
                        inputs = processor(st.session_state.current_image, return_tensors="pt")
                        
                        # Generate caption with different parameters based on model
                        with torch.no_grad():
                            if model_choice == "Large Model":
                                # Use more beams and longer max length for large model
                                out = model.generate(**inputs, max_length=100, num_beams=8, early_stopping=True)
                            else:
                                # Use fewer beams for base model for faster inference
                                out = model.generate(**inputs, max_length=50, num_beams=5)
                        
                        # Decode the caption
                        caption = processor.decode(out[0], skip_special_tokens=True)
                        
                        # Store in session state
                        st.session_state.original_caption = caption
                        st.session_state.enhanced_captions = []  # Clear previous enhanced captions
                        
                        # Display the caption with enhanced success message
                        st.success(f"🎉 Caption generated successfully using {model_choice}!")
                        st.balloons()  # Add celebration animation
                        st.write(f"**Original Caption:** {caption}")
                        
                        # Show caption statistics
                        word_count = len(caption.split())
                        st.info(f"📊 Caption contains {word_count} words")
                        
                        # Optional: Show raw output
                        with st.expander("Show technical details"):
                            st.code(f"Model: {selected_model}")
                            st.code(f"Raw model output: {out[0].tolist()}")
                            st.code(f"Decoded tokens: {caption}")
                            st.code(f"Caption length: {len(caption.split())} words")
                            
                    except Exception as e:
                        st.error(f"Error generating caption: {str(e)}")
                        st.error("Please try with a different image or check the image format.")
            
            # Display stored caption if available
            if st.session_state.original_caption:
                st.write(f"**Current Caption:** {st.session_state.original_caption}")
    
    with tab2:
        # Enhanced Captions Tab
        st.subheader("✨ Enhanced Captions")
        
        if st.session_state.original_caption is None:
            st.info("👆 Please generate a caption first in the 'Image & Caption' tab!")
        else:
            st.write(f"**Original Caption:** {st.session_state.original_caption}")
            st.markdown("---")
            
            if text_generator is not None:
                if st.button("Generate Enhanced Captions", type="primary"):
                    with st.spinner("✨ Generating enhanced captions..."):
                        enhanced_captions = enhance_caption(st.session_state.original_caption, text_generator)
                        
                        if enhanced_captions:
                            st.session_state.enhanced_captions = enhanced_captions
                            st.success(f"🎉 Generated {len(enhanced_captions)} enhanced captions!")
                            st.balloons()  # Add celebration animation
                        else:
                            st.warning("⚠️ Could not generate enhanced captions. Please try again.")
                
                # Display enhanced captions
                if st.session_state.enhanced_captions:
                    st.success(f"Found {len(st.session_state.enhanced_captions)} enhanced captions!")
                    
                    for i, enhanced_caption in enumerate(st.session_state.enhanced_captions, 1):
                        st.write(f"**Enhanced Caption {i}:** {enhanced_caption}")
                        st.markdown("---")
                    
                    # Download button for enhanced captions
                    st.markdown("---")
                    st.subheader("💾 Download Enhanced Captions")
                    
                    # Create enhanced captions content for download
                    enhanced_content = f"Enhanced Captions for Image\n"
                    enhanced_content += f"Original Caption: {st.session_state.original_caption}\n\n"
                    enhanced_content += f"Enhanced Captions:\n"
                    
                    for i, enhanced_caption in enumerate(st.session_state.enhanced_captions, 1):
                        enhanced_content += f"{i}. {enhanced_caption}\n"
                    
                    enhanced_content += f"\nGenerated with: {model_choice} ({selected_model})\n"
                    enhanced_content += f"Total enhanced captions: {len(st.session_state.enhanced_captions)}\n"
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Enhanced Captions",
                        data=enhanced_content,
                        file_name="enhanced_captions.txt",
                        mime="text/plain",
                        type="primary"
                    )
                    
                    # Preview content
                    with st.expander("Preview Enhanced Captions Content"):
                        st.text(enhanced_content)
                else:
                    st.info("Click 'Generate Enhanced Captions' to create creative variations!")
            else:
                st.warning("Text generator is not available. Enhanced captions cannot be generated.")
    
    with tab3:
        # Stats Tab
        st.subheader("📊 Caption Statistics")
        
        if st.session_state.original_caption is None:
            st.info("👆 Please generate a caption first in the 'Image & Caption' tab!")
        else:
            # Calculate statistics
            caption = st.session_state.original_caption
            word_count = len(caption.split())
            reading_time = word_count / 200  # Assuming 200 words per minute reading speed
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Caption Length", f"{word_count} words")
            
            with col2:
                st.metric("Reading Time", f"{reading_time:.1f} minutes")
            
            with col3:
                st.metric("Enhanced Captions", f"{len(st.session_state.enhanced_captions)}")
            
            st.markdown("---")
            
            # Download functionality
            st.subheader("💾 Download Captions")
            
            # Create text content for download
            download_content = f"Original Caption:\n{caption}\n\n"
            
            if st.session_state.enhanced_captions:
                download_content += "Enhanced Captions:\n"
                for i, enhanced_caption in enumerate(st.session_state.enhanced_captions, 1):
                    download_content += f"{i}. {enhanced_caption}\n"
                download_content += "\n"
            
            download_content += f"Statistics:\n"
            download_content += f"- Word count: {word_count}\n"
            download_content += f"- Reading time: {reading_time:.1f} minutes\n"
            download_content += f"- Enhanced captions: {len(st.session_state.enhanced_captions)}\n"
            download_content += f"- Generated with: {model_choice} ({selected_model})\n"
            
            # Download button
            st.download_button(
                label="📥 Download Captions as Text File",
                data=download_content,
                file_name="image_captions.txt",
                mime="text/plain",
                type="primary"
            )
            
            # Display raw content for preview
            with st.expander("Preview Download Content"):
                st.text(download_content)

else:
    # Instructions when no image is uploaded
    st.warning("⚠️ No image uploaded! Please upload an image to get started.")
    
    # Add some example information
    st.markdown("""
    ### How to use:
    1. **Select a model** from the sidebar (Base or Large)
    2. **Upload an image** using the file uploader above
    3. **Navigate through tabs** to generate and enhance captions
    4. **View statistics** and download your results
    
    ### Tab Overview:
    - **📸 Image & Caption**: Upload image and generate basic captions
    - **✨ Enhanced Captions**: Create creative variations using GPT-2
    - **📊 Stats**: View metrics and download captions as text file
    
    ### Model Options:
    - **Base Model**: Faster inference, lower memory usage, good for quick captions
    - **Large Model**: More detailed captions, better accuracy, slower but more comprehensive
    
    ### Features:
    - **Image Captioning**: Generate descriptive captions using BLIP models
    - **Caption Enhancement**: Use GPT-2 to create creative variations of captions
    - **Statistics**: Word count, reading time, and download functionality
    - **Model Selection**: Choose between base and large models based on your needs
    - **Dark Theme**: Beautiful dark interface for better user experience
    
    ### Supported formats:
    - JPG/JPEG images
    - PNG images
    
    ### About BLIP:
    The BLIP (Bootstrapping Language-Image Pre-training) model is designed to understand and describe images in natural language. It's particularly good at:
    - Describing objects and scenes
    - Understanding spatial relationships
    - Generating human-like descriptions
    - Adapting to different image types and contexts
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and HuggingFace Transformers")

