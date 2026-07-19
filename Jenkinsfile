pipeline {
    agent any

    environment {
        PYTHON = 'python3'
        VENV   = '.venv'
    }

    stages {

        stage('Checkout') {
            steps {
                echo 'SP-13: Recuperation du code depuis GitHub...'
                checkout scm
            }
        }

        stage('Setup Environment') {
            steps {
                echo 'Installation des dependances Python...'
                sh '''
                    python3 -m venv ${VENV}
                    . ${VENV}/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('DVC Pull') {
            steps {
                echo 'Recuperation des donnees avec DVC...'
                sh '''
                    . ${VENV}/bin/activate
                    dvc pull || echo "Pas de remote DVC configure - donnees locales utilisees"
                '''
            }
        }

        stage('Train Model') {
            steps {
                echo 'Entrainement du modele ML...'
                sh '''
                    . ${VENV}/bin/activate
                    python src/train.py
                '''
            }
        }

        stage('Validate Model') {
            steps {
                echo 'Validation du modele - seuil F1 >= 75%...'
                sh '''
                    . ${VENV}/bin/activate
                    python src/validate.py
                '''
            }
        }

        stage('Run Tests') {
            steps {
                echo 'Lancement des tests pytest...'
                sh '''
                    . ${VENV}/bin/activate
                    python -m pytest src/test_pipeline.py -v --tb=short
                '''
            }
        }

        stage('Docker Build') {
            steps {
                echo 'Construction de l image Docker...'
                sh 'docker build -t mlops-pipeline:latest .'
            }
        }

        stage('Deploy') {
            steps {
                echo 'Deploiement avec Docker Compose...'
                sh '''
                    docker compose down || true
                    docker compose up -d
                '''
            }
        }
    }

    post {
        success {
            echo 'Pipeline termine avec succes!'
        }
        failure {
            echo 'Pipeline echoue'
        }
    }
}
