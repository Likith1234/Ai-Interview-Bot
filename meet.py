import os
import re
import time
import tempfile
import pandas as pd
import openai
import speech_recognition as sr
from google.cloud import texttospeech
from google.cloud import speech_v1p1beta1 as speech
from IPython.display import Audio, display
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.units import inch

# Selenium and WebDriver imports
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

class InterviewAutomation:
    def __init__(self, dataframe_path, openai_api_key, email, password, meet_link, google_cloud_key_path):
        """
        Initialize the interview automation system.

        :param dataframe_path: Path to the CSV file containing interview questions and answers
        :param openai_api_key: OpenAI API key for response evaluation
        :param email: Google account email
        :param password: Google account password
        :param meet_link: Google Meet link
        :param google_cloud_key_path: Path to the Google Cloud service account key file
        """
        # OpenAI and Interview Setup
        self.dataframe_path = dataframe_path
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key
        self.recognizer = sr.Recognizer()
        self.expected_answers = []

        # Google Cloud Setup
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_cloud_key_path
        self.tts_client = texttospeech.TextToSpeechClient()
        self.stt_client = speech.SpeechClient()

        # WebDriver Setup
        self.email = email
        self.password = password
        self.meet_link = meet_link
        self.driver = self._setup_chrome_driver()

    def _setup_chrome_driver(self):
        """
        Setup and configure Chrome WebDriver.

        :return: Configured Chrome WebDriver
        """
        # Chrome options
        chrome_options = Options()
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)

        # Microphone and camera settings
        chrome_options.add_experimental_option("prefs", {
            "profile.default_content_setting_values.media_stream_mic": 1,
            "profile.default_content_setting_values.media_stream_camera": 1,
        })

        # WebDriver setup with automatic driver management
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver

    def google_login(self):
        """
        Perform Google account login with comprehensive error handling and debugging.
        """
        try:
            # Navigate to Google login
            self.driver.get('https://accounts.google.com/ServiceLogin')
            print("Navigated to Google login page")

            # Wait and find email input (with multiple possible locators)
            email_locators = [
                (By.ID, 'identifierId'),
                (By.NAME, 'identifier'),
                (By.XPATH, "//input[@type='email']")
            ]

            email_input = None
            for locator in email_locators:
                try:
                    email_input = WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located(locator)
                    )
                    break
                except Exception as e:
                    print(f"Failed to find email input with {locator}: {e}")

            if not email_input:
                raise Exception("Could not find email input field")

            # Enter email
            email_input.clear()
            email_input.send_keys(self.email)
            print(f"Entered email: {self.email}")

            # Find and click next button
            next_locators = [
                (By.ID, 'identifierNext'),
                (By.XPATH, "//button[@type='submit']"),
                (By.CSS_SELECTOR, "button[jsname='Cuz2Ue']")
            ]

            next_button = None
            for locator in next_locators:
                try:
                    next_button = WebDriverWait(self.driver, 10).until(
                        EC.element_to_be_clickable(locator)
                    )
                    next_button.click()
                    break
                except Exception as e:
                    print(f"Failed to click next button with {locator}: {e}")

            if not next_button:
                raise Exception("Could not find or click next button")

            # Wait for password input
            password_locators = [
                (By.NAME, 'password'),
                (By.XPATH, "//input[@type='password']"),
                (By.ID, 'password'),
                (By.NAME, 'Psw')
            ]

            password_input = None
            for locator in password_locators:
                try:
                    password_input = WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located(locator)
                    )
                    break
                except Exception as e:
                    print(f"Failed to find password input with {locator}: {e}")

            if not password_input:
                # Take a screenshot for debugging
                self.driver.save_screenshot("login_debug.png")
                raise Exception("Could not find password input field")

            # Enter password
            password_input.clear()
            password_input.send_keys(self.password)
            print("Entered password")

            # Find and click password next/submit button
            password_next_locators = [
                (By.ID, 'passwordNext'),
                (By.XPATH, "//button[@type='submit']"),
                (By.CSS_SELECTOR, "button[jsname='Cuz2Ue']")
            ]

            password_next_button = None
            for locator in password_next_locators:
                try:
                    password_next_button = WebDriverWait(self.driver, 10).until(
                        EC.element_to_be_clickable(locator)
                    )
                    password_next_button.click()
                    break
                except Exception as e:
                    print(f"Failed to click password next button with {locator}: {e}")

            if not password_next_button:
                # Take a screenshot for debugging
                self.driver.save_screenshot("login_debug_password.png")
                raise Exception("Could not find or click password next button")

            # Wait for login to complete or check for potential challenges
            WebDriverWait(self.driver, 15).until(
                EC.any_of(
                    EC.url_contains('myaccount.google.com'),
                    EC.url_contains('meet.google.com'),
                    EC.presence_of_element_located((By.ID, 'challenge'))
                )
            )

            print("Google login successful")

        except Exception as e:
            print(f"Login failed: {e}")
            # Take a screenshot for debugging
            try:
                self.driver.save_screenshot("login_error.png")
            except:
                print("Could not save screenshot")
            raise

    def join_google_meet(self):
        """
        Join the Google Meet session with comprehensive debugging.
        """
        try:
            # Extensive logging
            print("Starting Google Meet join process")
            print(f"Meet Link: {self.meet_link}")

            # Navigate to the meet link
            self.driver.get(self.meet_link)
            print("Navigated to meet link")

            # Wait for potential page load
            time.sleep(5)

            # Check current URL and page source
            current_url = self.driver.current_url
            print(f"Current URL after navigation: {current_url}")

            # Take a screenshot for debugging
            self.driver.save_screenshot("meet_join_debug.png")

            # Check if redirected to login
            if 'accounts.google.com' in current_url:
                print("Redirected to login page. Attempting re-login.")
                self.google_login()

                # Try navigating to meet link again
                self.driver.get(self.meet_link)
                time.sleep(5)

            # Comprehensive join button locators
            join_button_locators = [
                (By.CSS_SELECTOR, 'div.uArJ5e.UQuaGc.Y5sE8d.uyXBBb.xKiqt'),
                (By.XPATH, "//div[contains(text(), 'Join meeting')]"),
                (By.XPATH, "//span[contains(text(), 'Join')]"),
                (By.XPATH, "//div[contains(@role, 'button') and contains(text(), 'Join')]"),
                (By.CSS_SELECTOR, 'button[data-meeting-join-button="true"]')
            ]

            # Try multiple join button strategies
            join_button = None
            for locator in join_button_locators:
                try:
                    join_button = WebDriverWait(self.driver, 10).until(
                        EC.element_to_be_clickable(locator)
                    )
                    print(f"Found join button with locator: {locator}")
                    join_button.click()
                    break
                except Exception as e:
                    print(f"Failed to click join button with {locator}: {e}")

            if not join_button:
                print("Could not find any join button")
                # Save page source for detailed debugging
                with open("meet_page_source.html", "w", encoding="utf-8") as f:
                    f.write(self.driver.page_source)
                raise Exception("No join button found")

            # Wait for meeting interface
            time.sleep(5)

            # Handle microphone and camera
            self._handle_meeting_permissions()

            print("Successfully joined Google Meet")

        except Exception as e:
            print(f"CRITICAL ERROR joining Google Meet: {e}")
            # Save comprehensive debug information
            try:
                self.driver.save_screenshot("meet_join_critical_error.png")
                with open("meet_error_page_source.html", "w", encoding="utf-8") as f:
                    f.write(self.driver.page_source)
            except Exception as debug_e:
                print(f"Debug screenshot/source save failed: {debug_e}")
            raise

    def _handle_meeting_permissions(self):
        """
        Handle meeting permissions and initial setup.
        """
        try:
            # Locators for camera and mic buttons
            permission_locators = [
                (By.XPATH, "//div[contains(@aria-label, 'Microphone')]"),
                (By.XPATH, "//div[contains(@aria-label, 'Camera')]"),
                (By.CSS_SELECTOR, 'div[aria-label*="Microphone"]'),
                (By.CSS_SELECTOR, 'div[aria-label*="Camera"]')
            ]

            # Try to disable camera and mic
            for locator in permission_locators:
                try:
                    buttons = self.driver.find_elements(*locator)
                    for button in buttons:
                        try:
                            button.click()
                            print(f"Clicked permission button: {locator}")
                        except:
                            pass
                except Exception as e:
                    print(f"Failed to handle permissions with {locator}: {e}")

        except Exception as e:
            print(f"Permission handling error: {e}")

    def fetch_questions_and_answers(self):
        """
        Fetch questions and answers from the DataFrame.

        :return: List of questions and answers
        """
        try:
            # Verify if the file exists
            if not os.path.isfile(self.dataframe_path):
                raise FileNotFoundError(f"File not found: {self.dataframe_path}")

            df = pd.read_csv(self.dataframe_path)
            questions = df['Question'].tolist()
            answers = df['Answer'].tolist()
            print("Fetched Questions and Answers:", questions, answers)  # Debugging statement
            return questions, answers
        except Exception as e:
            print(f"Error fetching questions and answers: {e}")
            return [], []

    def text_to_speech(self, text, lang='en-US'):
        """
        Convert text to speech using Google Cloud Text-to-Speech and play it.

        :param text: Text to convert
        :param lang: Language of the text
        """
        try:
            # Set the text input to be synthesized
            synthesis_input = texttospeech.SynthesisInput(text=text)

            # Build the voice request, select the language code ("en-US") and the ssml voice gender ("neutral")
            voice = texttospeech.VoiceSelectionParams(language_code=lang, ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)

            # Select the type of audio file you want returned
            audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

            # Perform the text-to-speech request
            response = self.tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

            # The response's audio_content is binary
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
                temp_audio_file.write(response.audio_content)
                audio_file_path = temp_audio_file.name

            # Play the audio file using pygame
            pygame.mixer.init()
            pygame.mixer.music.load(audio_file_path)
            pygame.mixer.music.play()

            # Wait for the audio to finish playing
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            print(f"Playing audio: {audio_file_path}")

            # Clean up the temporary file
            os.remove(audio_file_path)

        except Exception as e:
            print(f"Text-to-speech error: {e}")
            print(text)

    def speech_to_text(self):
        """
        Capture speech input and convert to text using Google Cloud Speech-to-Text.

        :return: Transcribed text or empty string
        """
        try:
            recognizer = sr.Recognizer()
            recognizer.energy_threshold = 4000
            recognizer.pause_threshold = 2.5
            recognizer.non_speaking_duration = 1.5

            with sr.Microphone() as source:
                print("Listening for the answer...")
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source, timeout=10)

            # Save the audio to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
                temp_audio_file_path = temp_audio_file.name
                with open(temp_audio_file_path, "wb") as f:
                    f.write(audio.get_wav_data())

            # Read the audio file and send it to Google Cloud Speech-to-Text
            with open(temp_audio_file_path, "rb") as audio_file:
                content = audio_file.read()

            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="en-US",
            )

            response = self.stt_client.recognize(config=config, audio=audio)

            for result in response.results:
                print("Transcript: {}".format(result.alternatives[0].transcript))
                return result.alternatives[0].transcript

            return ""

        except Exception as e:
            print(f"Speech-to-text error: {e}")
            return ""

    def get_embeddings(self, text, tokenizer, model):
        inputs = tokenizer(text, return_tensors='pt')
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()

    def score_response(self, user_response, expected_answer, tokenizer, model):
        user_embedding = self.get_embeddings(user_response, tokenizer, model)
        expected_embedding = self.get_embeddings(expected_answer, tokenizer, model)
        similarity = cosine_similarity(user_embedding, expected_embedding)
        return similarity[0][0]

    def generate_interview_report(self, questions, user_responses, scores):
        """
        Generate a PDF report of the interview process.

        :param questions: List of questions asked
        :param user_responses: List of user responses
        :param scores: List of scores for each response
        """
        try:
            pdf_path = "interview_report.pdf"
            doc = SimpleDocTemplate(pdf_path, pagesize=letter)
            elements = []
            styles = getSampleStyleSheet()

            # Title
            elements.append(Paragraph("Interview Report", styles['Title']))
            elements.append(Spacer(1, 12))

            # Subtitle
            elements.append(Paragraph("Generated on: " + time.strftime("%Y-%m-%d %H:%M:%S"), styles['Normal']))
            elements.append(Spacer(1, 12))

            # Content
            for i, (question, response, score) in enumerate(zip(questions, user_responses, scores), 1):
                elements.append(Paragraph(f"Question {i}: {question}", styles['Heading2']))
                elements.append(Paragraph("User Response: " + response, styles['BodyText']))
                elements.append(Paragraph(f"Score: {score:.2f}/100", styles['BodyText']))
                elements.append(Spacer(1, 12))

            # Total Score
            total_score = sum(scores) / len(scores) if scores else 0
            elements.append(Paragraph(f"Total Score: {total_score:.2f}/100", styles['Heading3']))

            # Build the PDF
            doc.build(elements)
            print(f"Interview report generated: {pdf_path}")

        except Exception as e:
            print(f"Error generating interview report: {e}")

    def conduct_interview(self):
        """
        Conduct the entire interview process with explicit Google Meet navigation.
        """
        try:
            # Perform Google login
            self.google_login()
            print("Login successful. Attempting to join Google Meet.")

            # Navigate to Google Meet directly
            self.driver.get('https://meet.google.com')
            print("Navigated to Google Meet main page")

            # Wait a moment
            time.sleep(3)

            # Navigate to specific meet link
            self.driver.get(self.meet_link)
            print(f"Navigated to specific meet link: {self.meet_link}")

            # Wait for page to load
            time.sleep(5)

            # Attempt to join meeting
            join_button_locators = [
                (By.CSS_SELECTOR, 'div.uArJ5e.UQuaGc.Y5sE8d.uyXBBb.xKiqt'),
                (By.XPATH, "//div[contains(text(), 'Join meeting')]"),
                (By.XPATH, "//span[contains(text(), 'Join')]"),
                (By.XPATH, "//div[contains(@role, 'button') and contains(text(), 'Join')]"),
                (By.CSS_SELECTOR, 'button[data-meeting-join-button="true"]')
            ]

            for locator in join_button_locators:
                try:
                    join_button = WebDriverWait(self.driver, 10).until(
                        EC.element_to_be_clickable(locator)
                    )
                    join_button.click()
                    print(f"Clicked join button with locator: {locator}")
                    break
                except Exception as e:
                    print(f"Failed to click join button with {locator}: {e}")

            # Wait for meeting to load
            time.sleep(5)

            # Fetch questions and answers from DataFrame
            questions, self.expected_answers = self.fetch_questions_and_answers()
            total_score = 0

            if not questions:
                print("No questions found in the DataFrame.")
                return

            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained('bert-base-uncased')

            scores = []
            user_responses = []
            for i, (question, expected_answer) in enumerate(zip(questions, self.expected_answers), 1):
                print(f"Question {i}: {question}")

                # Read question using text-to-speech
                self.text_to_speech(question)

                # Wait for response
                response = self.speech_to_text()

                if response:
                    # Score the response
                    score = self.score_response(response, expected_answer, tokenizer, model)
                    score_percentage = int(score * 100)
                    scores.append(score_percentage)
                    total_score += score_percentage
                    user_responses.append(response)
                    print(f"Score for Question {i}: {score_percentage}/100")
                else:
                    scores.append(0)
                    user_responses.append("No response captured.")
                    print("No response captured.")

                print("-" * 50)
                time.sleep(2)  # Brief pause between questions

            # Calculate and announce final score
            marks = total_score / len(questions)
            print(f"Interview complete. Your total score is {marks:.2f}/100.")
            final_message = "Thank you for attending the interview today. We appreciate your time and the insights you shared with us.The interview process has now concluded. You may leave the meeting at this time. Further updates regarding the next steps will be communicated to you by our HR team.We wish you all the best and look forward to being in touch soon."
            print(final_message)
            self.text_to_speech(final_message)

            # Generate interview report
            self.generate_interview_report(questions, user_responses, scores)

        except Exception as e:
            print(f"Interview process error: {e}")
            # Optional: save screenshot for debugging
            try:
                self.driver.save_screenshot("interview_error.png")
            except:
                pass
        finally:
            # Close the browser
            self.driver.quit()

def main():
    # Configuration - replace with your actual values
    DATAFRAME_PATH = "questions_and_answers.csv"  # Ensure this path is correct
    OPENAI_API_KEY = "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"  # Replace with your actual OpenAI API key
    GOOGLE_EMAIL = "dummymail11108@gmail.com"
    GOOGLE_PASSWORD = "Likith@1234"
    MEET_LINK = "https://meet.google.com/wyw-dnux-bns"
    GOOGLE_CLOUD_KEY_PATH = "path/to/your/google-cloud-key.json"  # Replace with the path to your Google Cloud service account key file

    try:
        interview = InterviewAutomation(
            dataframe_path=DATAFRAME_PATH,
            openai_api_key=OPENAI_API_KEY,
            email=GOOGLE_EMAIL,
            password=GOOGLE_PASSWORD,
            meet_link=MEET_LINK,
            google_cloud_key_path=GOOGLE_CLOUD_KEY_PATH
        )
        interview.conduct_interview()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
