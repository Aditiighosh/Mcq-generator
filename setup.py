from setuptools import setup,find_packages


setup(
    name ="mcqgenerator" ,
    version = "0.0.1", 
    author = "Prateek Sharma" ,
    author_email = "sharma.prateek00000@gmail.com" , 
    install_requires = ["openai" , "langchain" , "streamlit", "python-dotenv" , "PyPDF2"] , 
    packages = find_packages() 
)