#!/usr/bin/env python3
"""
Inicializador de MySQL para Surf Consciente API
Crea la base de datos, tablas y datos iniciales si es necesario
"""

import sys
import os
from pathlib import Path

# Añadir el directorio raíz al path para importar módulos
sys.path.append(str(Path(__file__).parent.parent))

import logging
import pymysql
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError, OperationalError

from app.core.config import settings
from app.core.database import (
    engine, 
    create_database_if_not_exists,
    create_tables,
    check_database_connection,
    get_database_info,
    test_database_operations,
    get_db_context
)

# Configurar logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_mysql_server():
    """Verificar que el servidor MySQL esté ejecutándose"""
    try:
        connection = pymysql.connect(
            host=settings.MYSQL_HOST,
            port=settings.MYSQL_PORT,
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASSWORD,
            connect_timeout=10
        )
        connection.close()
        logger.info("✅ MySQL server is running and accessible")
        return True
    except pymysql.Error as e:
        logger.error(f"❌ MySQL server connection failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error connecting to MySQL: {e}")
        return False

def verify_mysql_credentials():
    """Verificar que las credenciales de MySQL sean correctas"""
    try:
        connection = pymysql.connect(
            host=settings.MYSQL_HOST,
            port=settings.MYSQL_PORT,
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASSWORD
        )
        
        with connection.cursor() as cursor:
            cursor.execute("SELECT USER(), @@version")
            user, version = cursor.fetchone()
            logger.info(f"✅ MySQL credentials verified - User: {user}, Version: {version}")
        
        connection.close()
        return True
        
    except pymysql.Error as e:
        logger.error(f"❌ MySQL credentials verification failed: {e}")
        return False

def create_user_if_not_exists():
    """Crear usuario de MySQL si no existe (solo si usamos root)"""
    if settings.MYSQL_USER == "root":
        logger.info("Using root user, skipping user creation")
        return True
    
    try:
        # Conectar como root para crear usuario
        root_connection = pymysql.connect(
            host=settings.MYSQL_HOST,
            port=settings.MYSQL_PORT,
            user="root",
            password="root"  # Cambiar según tu configuración
        )
        
        with root_connection.cursor() as cursor:
            # Crear usuario si no existe
            cursor.execute(f"""
                CREATE USER IF NOT EXISTS '{settings.MYSQL_USER}'@'%' 
                IDENTIFIED BY '{settings.MYSQL_PASSWORD}'
            """)
            
            # Otorgar permisos
            cursor.execute(f"""
                GRANT ALL PRIVILEGES ON {settings.MYSQL_DB}.* 
                TO '{settings.MYSQL_USER}'@'%'
            """)
            
            cursor.execute("FLUSH PRIVILEGES")
            
        root_connection.commit()
        root_connection.close()
        logger.info(f"✅ User '{settings.MYSQL_USER}' created/verified successfully")
        return True
        
    except pymysql.Error as e:
        logger.warning(f"⚠️ Could not create user (might already exist): {e}")
        return True  # Continue anyway
    except Exception as e:
        logger.error(f"❌ Error creating MySQL user: {e}")
        return False

def initialize_database():
    """Inicializar la base de datos completa"""
    logger.info("🚀 Starting MySQL database initialization...")
    
    # Paso 1: Verificar servidor MySQL
    if not check_mysql_server():
        logger.error("❌ MySQL server is not accessible. Please check your MySQL installation.")
        return False
    
    # Paso 2: Verificar credenciales
    if not verify_mysql_credentials():
        logger.error("❌ MySQL credentials are incorrect. Please check your .env file.")
        return False
    
    # Paso 3: Crear usuario si es necesario
    if not create_user_if_not_exists():
        logger.error("❌ Could not create MySQL user.")
        return False
    
    # Paso 4: Crear base de datos
    if not create_database_if_not_exists():
        logger.error("❌ Could not create database.")
        return False
    
    # Paso 5: Verificar conexión con SQLAlchemy
    if not check_database_connection():
        logger.error("❌ SQLAlchemy connection failed.")
        return False
    
    # Paso 6: Crear tablas
    try:
        create_tables()
        logger.info("✅ Database tables created successfully")
    except Exception as e:
        logger.error(f"❌ Error creating tables: {e}")
        return False
    
    # Paso 7: Probar operaciones básicas
    if not test_database_operations():
        logger.error("❌ Database operations test failed.")
        return False
    
    # Paso 8: Mostrar información de la base de datos
    db_info = get_database_info()
    logger.info("📊 Database Information:")
    logger.info(f"   Type: {db_info['type']}")
    logger.info(f"   Host: {db_info['host']}:{db_info['port']}")
    logger.info(f"   Database: {db_info['database']}")
    logger.info(f"   User: {db_info['user']}")
    logger.info(f"   Connected: {db_info['connected']}")
    logger.info(f"   Version: {db_info.get('version', 'Unknown')}")
    logger.info(f"   Tables: {db_info.get('tables', [])}")
    
    logger.info("🎉 MySQL database initialization completed successfully!")
    return True

def create_sample_data():
    """Crear datos de ejemplo para pruebas"""
    logger.info("📝 Creating sample data...")
    
    try:
        with get_db_context() as db:
            # Ejemplo: crear tabla de usuarios de prueba
            db.execute(text("""
                CREATE TABLE IF NOT EXISTS sample_users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    email VARCHAR(100) UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # Insertar datos de ejemplo
            db.execute(text("""
                INSERT IGNORE INTO sample_users (username, email) VALUES
                ('surfer1', 'surfer1@example.com'),
                ('surfer2', 'surfer2@example.com'),
                ('admin', 'admin@surfconsciente.com')
            """))
            
            # Verificar datos insertados
            result = db.execute(text("SELECT COUNT(*) FROM sample_users"))
            count = result.fetchone()[0]
            
            logger.info(f"✅ Sample data created successfully. {count} users in sample_users table")
            return True
            
    except Exception as e:
        logger.error(f"❌ Error creating sample data: {e}")
        return False

def reset_database():
    """Reiniciar la base de datos (eliminar y recrear)"""
    logger.warning("⚠️ Resetting database - This will delete all data!")
    
    try:
        from core.database import drop_tables
        
        # Eliminar tablas existentes
        drop_tables()
        logger.info("🗑️ Tables dropped successfully")
        
        # Recrear tablas
        create_tables()
        logger.info("🔄 Tables recreated successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error resetting database: {e}")
        return False

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize MySQL database for Surf Consciente API")
    parser.add_argument("--reset", action="store_true", help="Reset database (delete all data)")
    parser.add_argument("--sample-data", action="store_true", help="Create sample data")
    parser.add_argument("--info", action="store_true", help="Show database information only")
    
    args = parser.parse_args()
    
    if args.info:
        # Solo mostrar información
        db_info = get_database_info()
        print("\n" + "="*50)
        print("DATABASE INFORMATION")
        print("="*50)
        for key, value in db_info.items():
            print(f"{key.upper()}: {value}")
        print("="*50)
        return
    
    if args.reset:
        # Reiniciar base de datos
        if input("Are you sure you want to reset the database? (yes/no): ").lower() == "yes":
            reset_database()
        else:
            print("Database reset cancelled.")
            return
    
    # Inicializar base de datos
    success = initialize_database()
    
    if success and args.sample_data:
        create_sample_data()
    
    if success:
        print("\n🎉 Database initialization completed successfully!")
        print("You can now start your FastAPI application with: python main.py")
    else:
        print("\n❌ Database initialization failed. Please check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()