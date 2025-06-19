from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError, OperationalError
import logging
from typing import Generator, Dict, Any
from contextlib import contextmanager
import time
from .config import settings

# Configurar logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)

def create_mysql_engine():
    """Crea el engine de MySQL con configuración optimizada"""
    
    try:
        engine = create_engine(
            settings.database_url,
            pool_size=settings.MYSQL_POOL_SIZE,
            max_overflow=settings.MYSQL_MAX_OVERFLOW,
            pool_pre_ping=True,
            pool_recycle=settings.MYSQL_POOL_RECYCLE,
            pool_timeout=settings.MYSQL_POOL_TIMEOUT,
            echo=settings.DEBUG,
            connect_args=settings.mysql_connect_args,
            # Configuraciones adicionales para MySQL
            isolation_level="READ_COMMITTED",
            future=True
        )
        
        # Event listeners para MySQL
        @event.listens_for(engine, "connect")
        def set_mysql_pragma(dbapi_connection, connection_record):
            """Configurar parámetros de sesión MySQL"""
            try:
                with dbapi_connection.cursor() as cursor:
                    # Configurar timezone
                    cursor.execute("SET time_zone = '+00:00'")
                    
                    # Configurar SQL mode
                    cursor.execute("SET sql_mode = 'STRICT_TRANS_TABLES,NO_ZERO_DATE,NO_ZERO_IN_DATE,ERROR_FOR_DIVISION_BY_ZERO'")
                    
                    # Configurar charset
                    cursor.execute("SET NAMES utf8mb4 COLLATE utf8mb4_unicode_ci")
                    
                    # Configurar autocommit
                    cursor.execute("SET autocommit = 0")
                    
                logger.debug("MySQL session configured successfully")
            except Exception as e:
                logger.error(f"Error configuring MySQL session: {e}")
        
        @event.listens_for(engine, "checkout")
        def receive_checkout(dbapi_connection, connection_record, connection_proxy):
            """Verificar conexión al tomar del pool"""
            try:
                with dbapi_connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
            except Exception as e:
                logger.warning(f"Connection checkout failed, invalidating: {e}")
                # Invalidar conexión defectuosa
                connection_record.invalidate(e)
                raise
        
        return engine
        
    except Exception as e:
        logger.error(f"Error creating MySQL engine: {e}")
        raise

# Crear el engine
engine = create_mysql_engine()

# Crear session factory con configuración optimizada
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False,
    class_=Session
)

# Crear base class para los modelos
Base = declarative_base()

def get_db() -> Generator[Session, None, None]:
    """
    Dependency para obtener sesión de base de datos con manejo robusto de errores
    """
    db = SessionLocal()
    try:
        yield db
    except SQLAlchemyError as e:
        logger.error(f"Database error: {e}")
        db.rollback()
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

@contextmanager
def get_db_context():
    """Context manager para uso directo de la base de datos"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        logger.error(f"Database context error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def create_database_if_not_exists():
    """Crear la base de datos si no existe"""
    import pymysql
    
    try:
        # Conectar sin especificar la base de datos
        connection = pymysql.connect(
            host=settings.MYSQL_HOST,
            port=settings.MYSQL_PORT,
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASSWORD,
            charset=settings.MYSQL_CHARSET
        )
        
        with connection.cursor() as cursor:
            # Crear base de datos si no existe
            cursor.execute(f"""
                CREATE DATABASE IF NOT EXISTS `{settings.MYSQL_DB}` 
                CHARACTER SET {settings.MYSQL_CHARSET} 
                COLLATE {settings.MYSQL_COLLATION}
            """)
            logger.info(f"Database '{settings.MYSQL_DB}' created or already exists")
        
        connection.commit()
        connection.close()
        return True
        
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        return False

def create_tables():
    """Crear todas las tablas en la base de datos"""
    try:
        # Primero crear la base de datos si no existe
        if not create_database_if_not_exists():
            raise Exception("Could not create database")
        
        # Crear tablas
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        
        # Verificar que las tablas se crearon
        with engine.connect() as conn:
            result = conn.execute(text("SHOW TABLES"))
            tables = [row[0] for row in result]
            logger.info(f"Created tables: {tables}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise

def drop_tables():
    """Eliminar todas las tablas (usar con cuidado)"""
    try:
        Base.metadata.drop_all(bind=engine)
        logger.info("Database tables dropped successfully")
        return True
    except Exception as e:
        logger.error(f"Error dropping database tables: {e}")
        raise

def check_database_connection() -> bool:
    """Verificar la conexión a MySQL"""
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1 as test"))
            test_value = result.fetchone()[0]
            
            if test_value == 1:
                logger.info("MySQL connection successful")
                return True
            else:
                logger.error("MySQL connection test failed")
                return False
                
    except OperationalError as e:
        logger.error(f"MySQL connection failed - Operational Error: {e}")
        return False
    except Exception as e:
        logger.error(f"MySQL connection failed - Unexpected Error: {e}")
        return False

def get_database_info() -> Dict[str, Any]:
    """Obtener información detallada sobre la base de datos MySQL"""
    info = {
        "type": "MySQL",
        "host": settings.MYSQL_HOST,
        "port": settings.MYSQL_PORT,
        "database": settings.MYSQL_DB,
        "user": settings.MYSQL_USER,
        "charset": settings.MYSQL_CHARSET,
        "collation": settings.MYSQL_COLLATION,
        "connected": False,
        "version": None,
        "tables": []
    }
    
    try:
        with engine.connect() as connection:
            # Verificar conexión
            connection.execute(text("SELECT 1"))
            info["connected"] = True
            
            # Obtener versión de MySQL
            result = connection.execute(text("SELECT VERSION()"))
            info["version"] = result.fetchone()[0]
            
            # Obtener lista de tablas
            result = connection.execute(text("SHOW TABLES"))
            info["tables"] = [row[0] for row in result]
            
            # Obtener información adicional
            result = connection.execute(text("""
                SELECT 
                    @@character_set_database as charset,
                    @@collation_database as collation,
                    @@time_zone as timezone,
                    @@sql_mode as sql_mode
            """))
            row = result.fetchone()
            info.update({
                "current_charset": row[0],
                "current_collation": row[1],
                "timezone": row[2],
                "sql_mode": row[3]
            })
            
    except Exception as e:
        logger.error(f"Error getting database info: {e}")
        info["error"] = str(e)
    
    return info

def test_database_operations():
    """Probar operaciones básicas en la base de datos"""
    try:
        with get_db_context() as db:
            # Crear tabla de prueba
            db.execute(text("""
                CREATE TABLE IF NOT EXISTS test_connection (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    message VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # Insertar dato de prueba
            db.execute(text("""
                INSERT INTO test_connection (message) 
                VALUES ('Connection test successful')
            """))
            
            # Leer datos
            result = db.execute(text("SELECT * FROM test_connection ORDER BY id DESC LIMIT 1"))
            row = result.fetchone()
            
            # Limpiar tabla de prueba
            db.execute(text("DROP TABLE test_connection"))
            
            logger.info(f"Database operations test successful: {row}")
            return True
            
    except Exception as e:
        logger.error(f"Database operations test failed: {e}")
        return False