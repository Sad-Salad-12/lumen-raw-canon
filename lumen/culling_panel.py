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

from . import culling, nl_edit
from .i18n import tr
from .scheduler import Activity as A


def labels():
    return {'': tr('未分析'), 'keep': tr('推荐保留'), 'review': tr('待复核'), 'reject': tr('废片候选')}


def filters():
    return [('all', tr('全部照片')), ('keep', tr('推荐保留')), ('review', tr('待复核')),
            ('reject', tr('废片候选')), ('', tr('未分析'))]


class Relay(QObject):
    progress = Signal(object)


class CullingDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.window = parent
        parent.cull_dialog = self
        self.running = False
        self.cancel_event = None
        self.setWindowTitle(tr('AI 选片'))
        self.resize(1080, 720)
        self.setMinimumSize(840, 620)
        layout = QVBoxLayout(self)
        title = QLabel(tr('让 Gemini 找出值得保留的照片'))
        title.setObjectName('section')
        layout.addWidget(title)
        notice = QLabel(tr('发送相机显影后的 1600 像素预览，不上传 RAW 原片。按批调用 Gemini API，会产生用量；原片不会被删除。'))
        notice.setWordWrap(True)
        notice.setObjectName('subtle')
        layout.addWidget(notice)
        row = QHBoxLayout()
        self.scope = QComboBox()
        self.scope.addItem(tr('全部照片'), 'all')
        self.scope.addItem(tr('选中照片'), 'selected')
        if len(parent.selected_paths()) > 1:
            self.scope.setCurrentIndex(1)
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
        layout.addLayout(row)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels([tr('照片'), tr('文件名'), tr('选片结果'), tr('评分'), tr('置信度'), tr('判断理由')])
        self.table.setIconSize(QSize(104, 68))
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        for col, width in enumerate([120, 160, 130, 60, 75]):
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
        self.close_button = QPushButton(tr('关闭'))
        self.close_button.clicked.connect(self.reject)
        row.addWidget(self.close_button)
        layout.addLayout(row)
        self.paths = list(parent.documents)
        self.rows = {path: index for index, path in enumerate(self.paths)}
        self.table.setRowCount(len(self.paths))
        for path in self.paths:
            self.update_row(path)
        self.undo_button.setEnabled(bool(parent.cull_history))

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
                       *self.manual_buttons):
            widget.setEnabled(not value)
        self.cancel_button.setEnabled(value)
        self.undo_button.setEnabled(not value and bool(self.window.cull_history))

    def start(self):
        parent = self.window
        paths = parent.selected_paths() if self.scope.currentData() == 'selected' else list(parent.documents)
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
                  record.get('reason', '')]
        for col, text in enumerate(values):
            item = QTableWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, path)
            if col == 0:
                item.setIcon(icon)
            item.setToolTip(record.get('reason', '') + ('\n' + record['error'] if record.get('error') else ''))
            self.table.setItem(row, col, item)
        self.table.setRowHeight(row, 76)
        self.apply_filter(path=path)

    def apply_filter(self, *_, path=None):
        if not hasattr(self, 'table'):
            return
        selected = self.filter.currentData()
        rows = [(path, self.rows[path])] if path else self.rows.items()
        for photo, row in rows:
            hide = selected != 'all' and culling.status(self.window.documents[photo].get('culling', {})) != selected
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
            self.accept()
            self.window.open_path(paths[0])


class CullingMixin:
    def init_culling(self):
        self.cull_history = deque(maxlen=20)
        self.cull_relay = Relay(self)
        self.cull_preferences = culling.options()

    def build_culling_controls(self, layout):
        row = QHBoxLayout()
        self.cull_button = self.button(tr('AI 选片…'), self.open_culling)
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
            hide = selected != 'all' and verdict != selected
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
