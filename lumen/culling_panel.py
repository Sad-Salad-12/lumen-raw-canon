"""AI culling controls, review table, manual decisions and independent undo."""
import copy
import threading
from collections import deque
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Qt, QSize, QSignalBlocker
from PySide6.QtGui import QColor, QBrush
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QLineEdit, QCheckBox, QProgressBar, QTableWidget,
    QTableWidgetItem, QAbstractItemView, QHeaderView, QMessageBox)

from . import culling, nl_edit, collection, model
from .i18n import tr
from .scheduler import Activity as A


def labels():
    return {'': tr('未分析'), 'keep': tr('推荐保留'), 'review': tr('待复核'), 'reject': tr('排除区')}


def filters():
    return [('all', tr('全部照片')), ('keep', tr('推荐保留')), ('review', tr('待复核')),
            ('reject', tr('排除区')), ('', tr('未分析')), ('remaining', tr('未排除照片'))]


class Relay(QObject):
    progress = Signal(object)


class CullingDialog(QDialog):
    def __init__(self, parent, embedded=False):
        super().__init__(parent)
        self.window = parent
        self.embedded = embedded
        if embedded:
            self.setWindowFlags(Qt.WindowType.Widget)
        parent.cull_dialog = self
        self.running = False
        self.cancel_event = None
        self.setWindowTitle(tr('照片集工作台'))
        self.resize(1080, 720)
        self.setMinimumSize(840, 620)
        layout = QVBoxLayout(self)
        title = QLabel(tr('照片集工作台 · 先筛选，再初调'))
        title.setObjectName('section')
        layout.addWidget(title)
        header = QHBoxLayout()
        header.addWidget(QLabel(tr('1 批量选片 → 2 排除区 → 3 保留片初调 → 导出 / 精修')), 1)
        self.import_button = QPushButton(tr('＋ 导入多张'))
        self.import_button.clicked.connect(parent.open_file)
        header.addWidget(self.import_button)
        self.save_button = QPushButton(tr('保存选片集'))
        self.save_button.clicked.connect(parent.save_album)
        header.addWidget(self.save_button)
        layout.addLayout(header)
        notice = QLabel(tr('发送相机显影后的 1600 像素预览，不上传 RAW 原片。按批调用 Gemini API，会产生用量；原片不会被删除。'))
        notice.setWordWrap(True)
        notice.setObjectName('subtle')
        layout.addWidget(notice)
        row = QHBoxLayout()
        self.scope = QComboBox()
        self.scope.addItem(tr('全部照片'), 'all')
        self.scope.addItem(tr('选中照片'), 'selected')
        # A shoot is the default unit, regardless of filmstrip selection.
        row.addWidget(self.scope)
        self.genre = QComboBox()
        for key, name in [('general', tr('通用')), ('portrait', tr('人像')), ('travel', tr('旅行 / 风景')),
                          ('action', tr('运动 / 抓拍')), ('product', tr('产品')), ('artistic', tr('艺术 / 实验'))]:
            self.genre.addItem(name, key)
        self.genre.setCurrentIndex(self.genre.findData(parent.cull_preferences['genre']))
        row.addWidget(self.genre)
        row.addWidget(QLabel(tr('置信度门槛')))
        self.threshold = QSpinBox()
        self.threshold.setRange(75, 99)
        self.threshold.setValue(round(parent.cull_preferences['confidence'] * 100))
        self.threshold.setSuffix('%')
        self.threshold.setToolTip(tr('低于门槛的结果进入待复核；废片候选至少需要 90% 置信度。'))
        row.addWidget(self.threshold)
        self.force = QCheckBox(tr('重新分析'))
        self.force.setToolTip(tr('默认复用未变化的结果；人工改判始终保留。'))
        row.addWidget(self.force)
        row.addStretch()
        self.settings_button = QPushButton(tr('AI 设置'))
        self.settings_button.clicked.connect(self.configure)
        row.addWidget(self.settings_button)
        layout.addLayout(row)
        self.criteria = QLineEdit()
        self.criteria.setMaxLength(2000)
        self.criteria.setText(parent.cull_preferences['criteria'])
        self.criteria.setPlaceholderText(tr('选片偏好，例如：保留孩子自然的表情，允许夜景噪点，优先清晰的眼睛'))
        layout.addWidget(self.criteria)
        self.model_label = QLabel('')
        self.model_label.setTextFormat(Qt.TextFormat.PlainText)
        self.model_label.setObjectName('subtle')
        self.model_label.setWordWrap(True)
        layout.addWidget(self.model_label)
        self.update_service()
        row = QHBoxLayout()
        self.start_button = QPushButton(tr('开始 AI 选片'))
        self.start_button.setObjectName('primary')
        self.start_button.clicked.connect(self.start)
        row.addWidget(self.start_button)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        row.addWidget(self.progress, 1)
        self.cancel_button = QPushButton(tr('停止分析'))
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel)
        row.addWidget(self.cancel_button)
        layout.addLayout(row)
        self.summary = QLabel(tr('相似连拍每组最多比较 6 张；不确定的焦点、曝光或表情留给你复核。双击照片可查看大图。'))
        self.summary.setTextFormat(Qt.TextFormat.PlainText)
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.collection_counts = QLabel('')
        self.collection_counts.setObjectName('section')
        layout.addWidget(self.collection_counts)
        workflow = QHBoxLayout()
        self.remove_button = QPushButton(tr('一键移除排除区'))
        self.remove_button.clicked.connect(self.remove_excluded)
        workflow.addWidget(self.remove_button)
        self.adjust_mode = QComboBox()
        self.adjust_mode.addItem(tr('本地轻度初调'), 'local')
        self.adjust_mode.addItem(tr('Gemini 整组统一风格'), 'gemini')
        workflow.addWidget(self.adjust_mode)
        self.adjust_strength = QSpinBox()
        self.adjust_strength.setRange(1, 100)
        self.adjust_strength.setValue(50)
        self.adjust_strength.setSuffix('%')
        workflow.addWidget(self.adjust_strength)
        self.adjust_button = QPushButton(tr('批量初调保留片'))
        self.adjust_button.clicked.connect(self.start_adjustment)
        workflow.addWidget(self.adjust_button)
        self.export_button = QPushButton(tr('导出保留片'))
        self.export_button.clicked.connect(self.export_keeps)
        workflow.addWidget(self.export_button)
        layout.addLayout(workflow)
        options = QHBoxLayout()
        self.adjust_instruction = QLineEdit()
        self.adjust_instruction.setMaxLength(2000)
        self.adjust_instruction.setPlaceholderText(tr('整组初调方向，例如：自然肤色、统一暖调、保留夜景气氛'))
        options.addWidget(self.adjust_instruction, 1)
        self.adjust_overwrite = QCheckBox(tr('包含已有调色'))
        options.addWidget(self.adjust_overwrite)
        layout.addLayout(options)
        row = QHBoxLayout()
        self.filter = QComboBox()
        for key, name in filters():
            self.filter.addItem(name, key)
        self.filter.currentIndexChanged.connect(self.apply_filter)
        row.addWidget(self.filter)
        row.addStretch()
        self.undo_button = QPushButton(tr('撤销选片操作'))
        self.undo_button.clicked.connect(self.undo)
        row.addWidget(self.undo_button)
        self.collection_undo_button = QPushButton(tr('撤销移除 / 初调'))
        self.collection_undo_button.clicked.connect(self.undo_collection)
        row.addWidget(self.collection_undo_button)
        layout.addLayout(row)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([tr('照片'), tr('文件名'), tr('选片结果'), tr('评分'), tr('置信度'), tr('初调状态'), tr('判断理由')])
        self.table.setIconSize(QSize(104, 68))
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        for col, width in enumerate([120, 160, 130, 60, 75, 90]):
            self.table.setColumnWidth(col, width)
        self.table.cellDoubleClicked.connect(lambda *_: self.view_photo())
        self.table.itemSelectionChanged.connect(self.show_reason)
        layout.addWidget(self.table, 1)
        self.detail = QLabel('')
        self.detail.setWordWrap(True)
        self.detail.setMinimumHeight(44)
        self.detail.setMaximumHeight(100)
        self.detail.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.detail)
        row = QHBoxLayout()
        self.manual_buttons = []
        for key in culling.STATUSES:
            button = QPushButton(labels()[key])
            button.clicked.connect(lambda checked=False, value=key: self.mark(value))
            self.manual_buttons.append(button)
            row.addWidget(button)
        button = QPushButton(tr('恢复 AI 判断'))
        button.clicked.connect(lambda: self.mark(None))
        self.manual_buttons.append(button)
        row.addWidget(button)
        row.addStretch()
        self.view_button = QPushButton(tr('查看大图'))
        self.view_button.clicked.connect(self.view_photo)
        row.addWidget(self.view_button)
        self.close_button = QPushButton(tr('单张精修') if embedded else tr('关闭'))
        self.close_button.clicked.connect(self.reject)
        row.addWidget(self.close_button)
        layout.addLayout(row)
        self.paths, self.rows, self.row_states = [], {}, {}
        self.adjust_mode.currentIndexChanged.connect(lambda _: self.refresh_counts())
        self.load_preferences()
        self.sync_documents()
        self.undo_button.setEnabled(bool(parent.cull_history))

    def load_preferences(self):
        prefs = self.window.cull_preferences
        self.genre.setCurrentIndex(self.genre.findData(prefs['genre']))
        self.criteria.setText(prefs['criteria'])
        self.threshold.setValue(round(prefs['confidence'] * 100))
        prefs = self.window.collection_preferences
        self.adjust_mode.setCurrentIndex(self.adjust_mode.findData(prefs['mode']))
        self.adjust_instruction.setText(prefs['instruction'])
        self.adjust_strength.setValue(round(prefs['strength'] * 100))
        self.adjust_overwrite.setChecked(prefs['overwrite'])

    def sync_documents(self):
        paths = list(self.window.documents)
        if paths != self.paths:
            self.paths = paths
            self.rows = {path: index for index, path in enumerate(paths)}
            self.row_states = {}
            self.table.setRowCount(0)
            self.table.setRowCount(len(paths))
        for path in paths:
            doc = self.window.documents[path]
            state = (doc.get('culling', {}), doc.get('initial_adjustment', {}), doc['edits'])
            if path not in self.row_states or self.row_states[path] != state:
                self.update_row(path)
                self.row_states[path] = state
        self.refresh_counts()

    def refresh_counts(self):
        counts = {k: sum(culling.status(d.get('culling', {})) == k for d in self.window.documents.values()) for k in culling.STATUSES}
        self.collection_counts.setText(tr('保留 {keep} · 待复核 {review} · 排除区 {reject} · 共 {total} 张', **counts, total=len(self.window.documents)))
        busy = self.running or self.window.work.busy() or self.window.nl_busy
        self.remove_button.setEnabled(counts['reject'] > 0 and not busy)
        self.adjust_button.setEnabled(counts['keep'] > 0 and not busy)
        self.export_button.setEnabled(counts['keep'] > 0 and not busy)
        self.start_button.setEnabled(bool(self.window.documents) and not busy)
        self.collection_undo_button.setEnabled(bool(self.window.collection_history) and not busy)
        self.adjust_instruction.setEnabled(self.adjust_mode.currentData() == 'gemini' and not busy)

    def configure(self):
        self.window.nl_configure()
        self.update_service()

    def service(self):
        settings = copy.deepcopy(self.window.nl_settings)
        # Culling always uses the separately configured Google service, even if
        # natural-language editing is currently using another provider.
        settings['provider'] = 'gemini'
        return nl_edit.active(settings)

    def update_service(self):
        service = self.service()
        self.model_label.setText(tr('选片模型：{model} · 每批最多 6 张 · 人工改判始终保留', model=service['model']))

    def set_running(self, value):
        self.running = value
        for widget in (self.scope, self.genre, self.threshold, self.force, self.criteria,
                       self.settings_button, self.start_button, self.view_button, self.close_button,
                       self.import_button, self.save_button, self.adjust_mode, self.adjust_strength,
                       self.adjust_instruction, self.adjust_overwrite,
                       *self.manual_buttons):
            widget.setEnabled(not value)
        self.cancel_button.setEnabled(value)
        self.undo_button.setEnabled(not value and bool(self.window.cull_history))
        self.refresh_counts()

    def start(self):
        parent = self.window
        paths = (self.selected() if self.embedded else parent.selected_paths()) if self.scope.currentData() == 'selected' else list(parent.documents)
        if not paths or parent.nl_busy or not parent.work.can_start(A.AI):
            return
        service = self.service()
        if not service['key'] or service['api'] != 'gemini':
            message = tr('AI 选片需要 Google Gemini 原生接口，请在 AI 设置中配置。') if service['api'] != 'gemini' else tr('请先在设置中填写 {label} 的 API Key。', label='Google Gemini')
            QMessageBox.warning(self, tr('AI 选片'), message)
            return
        config = culling.options(self.genre.currentData(), self.criteria.text(), self.threshold.value() / 100,
                                 self.force.isChecked())
        preferences = dict(config, force=False)
        if parent.cull_preferences != preferences:
            parent.cull_preferences = preferences
            parent.library_structure_dirty = True
        records = {path: copy.deepcopy(parent.documents[path].get('culling', {})) for path in paths}
        self.before = records
        self.cancel_event = threading.Event()
        self.prepared_count = 0
        self.completed_paths = set()
        self.total_paths = len(paths)
        parent.work.begin(A.AI)
        parent.cancel_detail()
        self.set_running(True)
        self.progress.setValue(0)
        parent.cull_relay.progress.connect(self.on_progress)

        def work():
            return culling.run(paths, records, service, config, self.cancel_event, parent.cull_relay.progress.emit)

        def done(report):
            self.finish()
            count = {key: sum(culling.status(parent.documents[p].get('culling', {})) == key for p in paths) for key in culling.STATUSES}
            text = tr('推荐 {keep} · 待复核 {review} · 废片候选 {reject} · API 请求 {requests} · 复用 {cached}',
                      **count, requests=report['requests'], cached=report['cached'])
            unanalysed = sum(not culling.status(parent.documents[p].get('culling', {})) for p in paths)
            if unanalysed:
                text += ' · ' + tr('未分析') + ' ' + str(unanalysed)
            if report['cancelled']:
                text += ' · ' + tr('已停止，已完成的结果保留。')
            elif report['errors']:
                text += ' · ' + tr('分析失败，需人工复核。') + ' ' + report['errors'][0][1][:180]
            self.summary.setText(text)
            self.progress.setValue(100 if not report['cancelled'] else self.progress.value())

        def failed(message):
            self.finish()
            self.summary.setText(tr('分析失败，需人工复核。') + ' ' + message)

        parent.job(work, done, failed, priority=10)

    def finish(self):
        self.window.cull_relay.progress.disconnect(self.on_progress)
        self.window.work.end(A.AI)
        after = {path: copy.deepcopy(self.window.documents[path].get('culling', {})) for path in self.before if path in self.window.documents}
        changed = {p: old for p, old in self.before.items() if p in after and old != after[p]}
        if changed:
            self.window.cull_history.append(changed)
        self.set_running(False)
        self.window.update_library_status()
        self.window.resume_after_ai()

    def on_progress(self, event):
        path = event.get('path')
        if path and path in self.window.documents:
            self.window.documents[path]['culling'] = copy.deepcopy(event['record'])
            self.completed_paths.add(path)
            self.window.refresh_culling(path=path)
            self.update_row(path)
        elif event.get('phase') in ('prepare', 'analyze'):
            phase = tr('正在准备选片预览') if event['phase'] == 'prepare' else tr('正在分析照片')
            self.summary.setText(f"{phase} · {event['done']} / {event['total']} · {event['name']}")
            if event['phase'] == 'prepare':
                self.prepared_count = max(self.prepared_count, event['done'])
        if self.running:
            # Decode and API phases interleave; progress must never move backwards.
            percent = round((45 * self.prepared_count + 55 * len(self.completed_paths)) / max(1, self.total_paths))
            self.progress.setValue(max(self.progress.value(), percent))

    def cancel(self):
        if self.cancel_event:
            self.cancel_event.set()
            self.cancel_button.setEnabled(False)
            self.summary.setText(tr('正在停止；当前请求结束后停止，不再发送下一批。'))

    def reject(self):
        if self.running:
            self.cancel()
            return
        if self.embedded:
            self.view_photo()
            return
        super().reject()

    def closeEvent(self, event):
        if self.running:
            self.cancel()
            event.ignore()
        else:
            super().closeEvent(event)

    def update_row(self, path):
        row = self.rows[path]
        record = self.window.documents[path].get('culling', {})
        verdict = culling.status(record)
        icon = self.window.film_item(path).icon()
        values = ['', Path(path).name, labels()[verdict] + (' · ' + tr('人工') if record.get('manual') else ''),
                  str(round(record['score'])) if 'score' in record and not record.get('error') else '—',
                  f"{record['confidence']:.0%}" if 'confidence' in record and not record.get('error') else '—',
                  self.adjustment_label(path), record.get('reason', '')]
        for col, text in enumerate(values):
            item = QTableWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, path)
            if col == 0:
                item.setIcon(icon)
            item.setToolTip(record.get('reason', '') + ('\n' + record['error'] if record.get('error') else ''))
            self.table.setItem(row, col, item)
        self.table.setRowHeight(row, 76)
        self.apply_filter(path=path)

    def adjustment_label(self, path):
        doc = self.window.documents[path]
        metadata = doc.get('initial_adjustment', {})
        if metadata.get('status') == 'adjusted':
            return tr('已初调')
        if metadata.get('status') == 'error':
            return tr('初调失败')
        return tr('已有调色') if collection.has_look(doc['edits']) else tr('待初调')

    def apply_filter(self, *_, path=None):
        if not hasattr(self, 'table'):
            return
        selected = self.filter.currentData()
        rows = [(path, self.rows[path])] if path else self.rows.items()
        for photo, row in rows:
            verdict = culling.status(self.window.documents[photo].get('culling', {}))
            hide = verdict == 'reject' if selected == 'remaining' else selected != 'all' and verdict != selected
            self.table.setRowHidden(row, hide)
            if hide:
                for col in range(self.table.columnCount()):
                    item = self.table.item(row, col)
                    if item:
                        item.setSelected(False)

    def selected(self):
        return [self.paths[index.row()] for index in self.table.selectionModel().selectedRows() if not self.table.isRowHidden(index.row())]

    def show_reason(self):
        paths = self.selected()
        if not paths:
            self.detail.clear()
            return
        record = self.window.documents[paths[0]].get('culling', {})
        text = record.get('reason', '')
        if record.get('duplicate_of'):
            text += '\n' + tr('较佳相似照片：{name}', name=record['duplicate_of'])
        if record.get('error'):
            text += '\n' + record['error']
        if record.get('manual'):
            text = tr('人工改判') + ' · ' + text
        if record.get('analyzed_at'):
            text += '\n' + tr('模型') + ': ' + record.get('model', '') + ' · ' + record['analyzed_at']
        if record.get('criteria'):
            text += '\n' + record['criteria']
        metadata = self.window.documents[paths[0]].get('initial_adjustment', {})
        if metadata.get('explanation'):
            text += '\n' + self.adjustment_label(paths[0]) + ': ' + metadata['explanation']
        self.detail.setText(text)

    def mark(self, verdict):
        self.window.mark_culling(self.selected(), verdict)
        for path in self.selected():
            self.update_row(path)
        self.undo_button.setEnabled(bool(self.window.cull_history))
        self.show_reason()

    def undo(self):
        self.window.undo_culling()
        for path in self.paths:
            self.update_row(path)
        self.undo_button.setEnabled(bool(self.window.cull_history))

    def view_photo(self):
        paths = self.selected()
        if paths and not self.running:
            if not self.embedded:self.accept()
            self.window.open_path(paths[0])

    def remove_excluded(self):
        self.window.remove_excluded()
        self.sync_documents()
        self.summary.setText(tr('排除区已移出图集；原片保留，可撤销移除。'))

    def undo_collection(self):
        self.window.undo_collection()
        self.sync_documents()

    def export_keeps(self):
        if self.window.work.busy() or self.window.nl_busy or not any(culling.status(d.get('culling', {})) == 'keep' for d in self.window.documents.values()):return
        self.window.select_culling_keeps()
        self.window.batch_export()

    def start_adjustment(self):
        parent = self.window
        if parent.nl_busy or not parent.work.can_start(A.AI):return
        parent.commit()
        parent.stash_document()
        documents = {p: copy.deepcopy(d) for p, d in parent.documents.items() if culling.status(d.get('culling', {})) == 'keep'}
        if not documents:return
        config = collection.settings(self.adjust_mode.currentData(), self.adjust_instruction.text(),
                                     self.adjust_strength.value() / 100, self.adjust_overwrite.isChecked())
        genre = self.genre.currentData()
        service = self.service()
        if config['mode'] == 'gemini' and (not service['key'] or service['api'] != 'gemini'):
            QMessageBox.warning(self, tr('照片集工作台'), tr('AI 选片需要 Google Gemini 原生接口，请在 AI 设置中配置。'))
            return
        if parent.collection_preferences != config:
            parent.collection_preferences = copy.deepcopy(config)
            parent.library_structure_dirty = True
        fields = ('edits', 'initialized', 'initial_adjustment', 'initial_base', 'history')
        before = {p: {k: copy.deepcopy(d[k]) for k in fields if k in d} for p, d in documents.items()}
        self.cancel_event = threading.Event()
        self.prepared_count = 0
        self.completed_paths = set()
        self.total_paths = len(documents)
        parent.work.begin(A.AI)
        parent.cancel_detail()
        self.set_running(True)
        self.progress.setValue(0)
        parent.cull_relay.progress.connect(self.on_adjustment_progress)

        def finish(report=None, error=''):
            parent.cull_relay.progress.disconnect(self.on_adjustment_progress)
            parent.work.end(A.AI)
            changed = {p: original for p, original in before.items() if any(parent.documents[p].get(k) != v for k, v in original.items() if k != 'history')}
            if changed:parent.collection_history.append(dict(kind='adjust', before=changed))
            self.set_running(False)
            parent.update_library_status()
            parent.resume_after_ai()
            if error:
                self.summary.setText(tr('初调失败') + ': ' + error)
                return
            done = sum(v.get('initial_adjustment', {}).get('status') == 'adjusted' for v in report['changes'].values())
            self.summary.setText(tr('初调 {done} · 复用 {cached} · 跳过已有调色 {skipped} · 失败 {failed} · API 请求 {requests}',
                                    done=done, cached=report['cached'], skipped=report['skipped'], failed=len(report['errors']), requests=report['requests'])
                                 + (' · ' + tr('已停止，已完成的结果保留。') if report['cancelled'] else ''))
            if not report['cancelled']:self.progress.setValue(100)

        parent.job(lambda: collection.run(documents, config, service, genre, self.cancel_event,
                                         parent.cull_relay.progress.emit), finish, lambda e: finish(error=e), priority=10)

    def on_adjustment_progress(self, event):
        path = event.get('path')
        if path and path in self.window.documents:
            value = copy.deepcopy(event['adjustment'])
            doc = self.window.documents[path]
            if 'edits' in value:
                doc.setdefault('history', model.History(doc['edits'])).push(value['edits'])
            doc.update(value)
            self.window.restore_current_document(path)
            self.completed_paths.add(path)
            self.update_row(path)
        else:
            self.summary.setText(tr('正在初调整组保留片') + f" · {event['done']} / {event['total']} · {event['name']}")
            percent = round(100 * event['done'] / max(1, event['total']))
            self.progress.setValue(max(self.progress.value(), percent))


class CullingMixin:
    def init_culling(self):
        self.cull_history = deque(maxlen=20)
        self.cull_relay = Relay(self)
        self.cull_preferences = culling.options()
        self.collection_preferences = collection.settings()
        self.collection_history = deque(maxlen=20)

    def build_culling_controls(self, layout):
        row = QHBoxLayout()
        self.cull_button = self.button(tr('照片集工作台'), self.open_culling)
        row.addWidget(self.cull_button)
        self.cull_filter = QComboBox()
        for key, name in filters():
            self.cull_filter.addItem(name, key)
        self.cull_filter.currentIndexChanged.connect(lambda *_: self.update_library_status())
        row.addWidget(self.cull_filter)
        self.cull_summary = QLabel('')
        self.cull_summary.setObjectName('subtle')
        row.addWidget(self.cull_summary)
        row.addStretch()
        self.cull_select_button = self.button(tr('选中推荐片'), self.select_culling_keeps)
        row.addWidget(self.cull_select_button)
        layout.addLayout(row)

    def open_culling(self):
        if not self.documents or self.work.busy() or self.nl_busy:
            return
        if hasattr(self, 'collection_page'):
            self.show_collection()
            return
        dialog = CullingDialog(self)
        try:
            dialog.exec()
        finally:
            self.cull_dialog = None
            dialog.deleteLater()

    def refresh_culling(self, *_, path=None):
        if not hasattr(self, 'cull_filter') or not hasattr(self, 'filmstrip'):
            return
        from .library import dirty
        counts = {key: 0 for key in ('', *culling.STATUSES)}
        for document in self.documents.values():
            counts[culling.status(document.get('culling', {}))] += 1
        selected = self.cull_filter.currentData()
        blocker = QSignalBlocker(self.filmstrip)
        items = ({path: self.film_item(path)} if path else {
            self.filmstrip.item(i).data(Qt.ItemDataRole.UserRole): self.filmstrip.item(i)
            for i in range(self.filmstrip.count())})
        for photo, item in items.items():
            document = self.documents.get(photo)
            if item is None or document is None:
                continue
            record = document.get('culling', {})
            verdict = culling.status(record)
            hide = verdict == 'reject' if selected == 'remaining' else selected != 'all' and verdict != selected
            item.setHidden(hide)
            if hide:
                item.setSelected(False)
            badge = {'keep': '✓', 'review': '?', 'reject': '×'}.get(verdict, '')
            item.setText((badge + ' ' if badge else '') + Path(photo).name + (' •' if dirty(document) else ''))
            item.setForeground(QBrush(QColor({'keep': '#c5d8b0', 'review': '#e6ca8d', 'reject': '#e2a79b'}.get(verdict, '#d9ded8'))))
            reason = labels()[verdict] + (' · ' + tr('人工') if record.get('manual') else '')
            item.setToolTip(photo + ('\n' + reason + '\n' + record.get('reason', '') if verdict else ''))
        del blocker
        self.cull_summary.setText(tr('推荐 {keep} · 待复核 {review} · 废片候选 {reject}', **{k: counts[k] for k in culling.STATUSES}))
        self.cull_button.setEnabled(bool(self.documents) and not self.work.busy() and not self.nl_busy)
        self.cull_select_button.setEnabled(counts['keep'] > 0 and not self.work.busy())

    def select_culling_keeps(self):
        if self.work.busy():
            return
        self.cull_filter.setCurrentIndex(1)
        self.filmstrip.clearSelection()
        for path, document in self.documents.items():
            if culling.status(document.get('culling', {})) == 'keep':
                self.film_item(path).setSelected(True)
        self.update_library_status()

    def mark_culling(self, paths, verdict):
        if self.work.busy() or verdict not in (*culling.STATUSES, None):
            return
        before = {}
        for path in paths:
            if path not in self.documents:
                continue
            old = copy.deepcopy(self.documents[path].get('culling', {}))
            new = copy.deepcopy(old)
            if verdict is None:
                new.pop('manual', None)
            else:
                new['manual'] = verdict
            if old != new:
                before[path] = old
                self.documents[path]['culling'] = new
        if before:
            self.cull_history.append(before)
            self.refresh_culling()
            self.update_library_status()

    def undo_culling(self):
        if self.work.busy() or not self.cull_history:
            return
        for path, record in self.cull_history.pop().items():
            if path in self.documents:
                self.documents[path]['culling'] = copy.deepcopy(record)
        self.refresh_culling()
        self.update_library_status()

    def show_collection(self):
        if not hasattr(self, 'workspace_stack'):return
        self.commit()
        self.stash_document()
        self.collection_page.sync_documents()
        self.workspace_stack.setCurrentWidget(self.collection_page)
        self.refresh_access()

    def restore_current_document(self, path):
        if path != self.source_path or self.source is None:return
        doc = self.documents[path]
        self.edits = copy.deepcopy(doc['edits'])
        self.saved_edits = copy.deepcopy(doc['saved_edits'])
        self.history = copy.deepcopy(doc.get('history', model.History(self.edits)))
        self.current_mask = min(self.current_mask, len(self.edits['masks']) - 1)
        self.nl_plan = self.nl_applied = None
        self.refresh()
        self.changed()
        self.history_timer.stop()

    def remove_excluded(self):
        if self.work.busy() or self.nl_busy:return
        self.stash_document()
        paths = [p for p, d in self.documents.items() if culling.status(d.get('culling', {})) == 'reject']
        if not paths:return
        order = list(self.documents)
        removed = {p: self.documents[p] for p in paths}
        icons = {p: self.film_item(p).icon() for p in paths}
        self.collection_history.append(dict(kind='remove', documents=removed, order=order, icons=icons))
        if self.source_path in removed:self.clear_current_photo()
        blocker = QSignalBlocker(self.filmstrip)
        for p in paths:
            item = self.film_item(p)
            self.filmstrip.takeItem(self.filmstrip.row(item))
            self.documents.pop(p)
        del blocker
        self.thumbnail_queue = [p for p in self.thumbnail_queue if p not in removed]
        self.library_structure_dirty = True
        self.update_library_status()
        self.statusBar().showMessage(tr('已从图集移除 {n_paths} 张 · 磁盘原片保留', n_paths=len(paths)))

    def undo_collection(self):
        if self.work.busy() or self.nl_busy or not self.collection_history:return
        self.stash_document()
        event = self.collection_history.pop()
        if event['kind'] == 'adjust':
            for p, values in event['before'].items():
                if p not in self.documents:continue
                self.documents[p].pop('history', None)
                self.documents[p].update(copy.deepcopy(values))
                self.restore_current_document(p)
        else:
            # Reinsert removed documents in their original order; newly imported
            # photographs remain, and explicitly reimported paths are not overwritten.
            current = dict(self.documents)
            restored = {p: d for p, d in event['documents'].items() if p not in current}
            current.update(restored)
            order = [p for p in event['order'] if p in current] + [p for p in current if p not in event['order']]
            icons = {p: self.film_item(p).icon() for p in self.documents}
            icons.update({p: event['icons'][p] for p in restored})
            blocker = QSignalBlocker(self.filmstrip)
            self.filmstrip.clear()
            self.documents = {}
            self.add_documents(order)
            self.documents = {p: current[p] for p in order}
            for p, icon in icons.items():self.film_item(p).setIcon(icon)
            del blocker
            self.library_structure_dirty = True
        self.update_library_status()
