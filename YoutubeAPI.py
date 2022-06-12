from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException, UnexpectedAlertPresentException
from selenium.webdriver.common.by import By
import time
import os
import logging


class YoutubeAPI:

    def __init__(self, *, user_data_dir, driverPath=None):

        if driverPath is None:
            driverPath = ChromeDriverManager().install()

        options = webdriver.ChromeOptions()
        options.add_argument(f"user-data-dir={user_data_dir}")
        options.add_experimental_option('excludeSwitches', ['enable-logging'])

        self.driver = webdriver.Chrome(
            service=Service(driverPath), options=options)

    def uploadVideo(self, *args, **kwargs):
        try:
            self._uploadVideo(*args, **kwargs)

        except Exception as e:
            raise e

        finally:
            self.driver.quit()
            pass

    def _uploadVideo(self, videoPath, title=None, description='', visibility='private', nsfw=False):
        if title == None:
            title = os.path.basename(videoPath)

        visibility = visibility.lower()

        self.driver.implicitly_wait(0.5)
        self.driver.maximize_window()

        while True:

            self.driver.get('https://youtube.com/upload')
            self.driver.implicitly_wait(1.5)

            logging.debug('Loaded Youtube Page')

            try:
                # Dismisses new content creator notifications
                self.driver.find_element(
                    By.XPATH, '//*[@id="dismiss-button"]').click()
                logging.debug('notification dismissed')
            except NoSuchElementException:
                logging.warning('element not found - dismiss button')
                pass
            # except UnexpectedAlertPresentException:
            #     logging.warning('UnexpectedAlertException triggered')

            logging.debug('Uploading started')
            self.driver.find_element(By.NAME, 'Filedata').send_keys(
                videoPath)  # Opens youtube and uploads the video
            logging.debug('Uploading finished')

            try:
                self.driver.implicitly_wait(8)
                titleInput = self.driver.find_element(
                    By.XPATH, "//div[@id='textbox' and @aria-label='Add a title that describes your video']")
                # titleInput = self.driver.find_element(
                #     By.XPATH, '/html/body/ytcp-uploads-dialog/tp-yt-paper-dialog/div/ytcp-animatable[1]/ytcp-video-metadata-editor/div/ytcp-video-metadata-editor-basics/div[1]/ytcp-social-suggestions-textbox/ytcp-form-input-container/div[1]/div[2]/div/ytcp-mention-input/div')
                logging.debug('TitleInput found')
            except NoSuchElementException:
                logging.warning(
                    'metadata editor not found, switching to window')
                self.driver.switch_to.window(self.driver.window_handles[0])
                continue
            else:
                break

        titleInput.clear()
        titleInput.send_keys(title)  # Enters title and description
        description = "#shorts\n" + description
        # self.driver.find_element(
        #     By.XPATH, '/html/body/ytcp-uploads-dialog/tp-yt-paper-dialog/div/ytcp-animatable[1]/ytcp-video-metadata-editor/div/ytcp-video-metadata-editor-basics/div[2]/ytcp-social-suggestions-textbox/ytcp-form-input-container/div[1]/div[2]/div/ytcp-mention-input/div').send_keys(description)

        self.driver.find_element(
            By.XPATH, "//div[@id='textbox' and @aria-label='Tell viewers about your video']").send_keys(description)

        if nsfw:
            self.driver.find_element(
                By.XPATH, '//*[@id="audience"]/ytkc-made-for-kids-select/div[4]/tp-yt-paper-radio-group/tp-yt-paper-radio-button[2]').click()
        else:
            # Sets whether video is made for kids
            self.driver.find_element(
                By.XPATH, '//*[@id="radioLabel"]/ytcp-ve').click()

        s = self.driver.find_element(By.XPATH, '//*[@id="next-button"]/div')

        s.click()
        s.click()  # Proceeds through youtube upload menu
        s.click()

        if visibility == 'private':
            self.driver.find_element(
                By.XPATH, '//*[@id="private-radio-button"]').click()

        elif visibility == 'unlisted':
            # Sets the visibility mode of the video
            self.driver.find_element(
                By.XPATH, '//*[@id="privacy-radios"]/tp-yt-paper-radio-button[2]').click()

        elif visibility == 'public':
            self.driver.find_element(
                By.XPATH, '//*[@id="privacy-radios"]/tp-yt-paper-radio-button[3]').click()

        else:
            assert False, "invalid visibility"  # buddy you can use the raise keyword

        while True:
            try:
                s = self.driver.find_element(
                    By.XPATH, '//*[@id="done-button"]')  # Publishes the video
                s.click()
                time.sleep(0.1)
            except:
                break

        time.sleep(3)
        logging.info('Video Upload Process Completed')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    running_dir = os.path.dirname(os.path.realpath(__file__))
    videoDir = os.path.join(running_dir, 'videos')
    videoPath = os.path.join(videoDir, os.listdir(videoDir)[-1])
    # videoPath = r'C:\Code\Random_Shit\RedditReaderV2\videos\video.mp4'

    for i in range(10):
        e = YoutubeAPI(
            user_data_dir=r"C:\Users\micha\AppData\Local\Google\Chrome\User Data")

        e.uploadVideo(
            videoPath, title=f'video {i}', description="not so poggers")
